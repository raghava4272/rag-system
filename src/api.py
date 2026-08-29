"""Typed HTTP boundary for the existing deterministic RAG pipeline."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, Request, Response, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from scripts.ask import build_phase2_retriever
from src.auth import APIKeyAuthenticator
from src.embed import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
)
from src.generate import GeneratedAnswer
from src.llm import DEFAULT_GROQ_MODEL, GroqLanguageModel
from src.observability import (
    bind_log_context,
    configure_application_logging,
    normalize_request_id,
)
from src.permissions import DocumentPermissions
from src.pipeline import RAGPipeline
from src.rate_limit import RateLimitDecision, TokenBucketRateLimiter
from src.rerank import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
)
from src.retrieve import DEFAULT_TOP_K


configure_application_logging()
LOGGER = logging.getLogger("rag.api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_QUESTION_LENGTH = 4_096
MAX_TOP_K = 20


class AnswerPipeline(Protocol):
    """The small application interface required by the HTTP layer."""

    def ask(
        self,
        question: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str],
    ) -> GeneratedAnswer: ...


PipelineFactory = Callable[[], AnswerPipeline]


class Authenticator(Protocol):
    def authenticate(self, api_key: str) -> str | None: ...


AuthenticatorFactory = Callable[[], Authenticator]


class PermissionResolver(Protocol):
    def allowed_sources(self, user_id: str) -> frozenset[str] | None: ...


PermissionResolverFactory = Callable[[], PermissionResolver]


class RateLimiter(Protocol):
    def check(self, identity: str) -> RateLimitDecision: ...


RateLimiterFactory = Callable[[], RateLimiter]


class QueryRequest(BaseModel):
    """Validated client input for one grounded question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class CitationResponse(BaseModel):
    number: int
    record_id: str
    source: str
    page_number: int | None
    chunk_index: int | None


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    user_id: str
    session_id: str
    citations: list[CitationResponse]


class HealthResponse(BaseModel):
    status: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def build_default_pipeline() -> RAGPipeline:
    """Create the production pipeline once from runtime configuration."""

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "The API requires python-dotenv. Install the locked dependencies."
        ) from error

    load_dotenv(PROJECT_ROOT / ".env")
    embedding_provider = SentenceTransformerEmbeddingProvider(
        os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    vector_store = ChromaVectorStore(
        path=Path(os.getenv("RAG_CHROMA_PATH", str(DEFAULT_CHROMA_PATH))),
        collection_name=os.getenv("RAG_COLLECTION", DEFAULT_COLLECTION_NAME),
    )
    retriever = build_phase2_retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker_model=os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        candidate_k=DEFAULT_RERANK_CANDIDATE_K,
        min_relevance_score=DEFAULT_MIN_RELEVANCE_SCORE,
    )
    return RAGPipeline(
        retriever=retriever,
        language_model=GroqLanguageModel(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        ),
    )


def build_default_authenticator() -> APIKeyAuthenticator:
    """Load hashed API-key identities from the runtime environment."""

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "The API requires python-dotenv. Install the locked dependencies."
        ) from error

    load_dotenv(PROJECT_ROOT / ".env")
    return APIKeyAuthenticator.from_environment()


def build_default_document_permissions() -> DocumentPermissions:
    """Load exact per-user document scopes from runtime configuration."""

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "The API requires python-dotenv. Install the locked dependencies."
        ) from error

    load_dotenv(PROJECT_ROOT / ".env")
    return DocumentPermissions.from_environment()


def build_default_rate_limiter() -> TokenBucketRateLimiter:
    """Load bounded per-user request limits from runtime configuration."""

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "The API requires python-dotenv. Install the locked dependencies."
        ) from error

    load_dotenv(PROJECT_ROOT / ".env")
    return TokenBucketRateLimiter.from_environment()


def _safe_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Cache-Control": "no-store"},
        content={
            "error": {
                "code": "rag_unavailable",
                "message": "The RAG service is temporarily unavailable.",
            }
        },
    )


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={
            "Cache-Control": "no-store",
            "WWW-Authenticate": "Bearer",
        },
        content={
            "error": {
                "code": "unauthorized",
                "message": "Valid bearer authentication is required.",
            }
        },
    )


def _forbidden_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        headers={"Cache-Control": "no-store"},
        content={
            "error": {
                "code": "forbidden",
                "message": "The authenticated user cannot access this resource.",
            }
        },
    )


def _rate_limit_headers(decision: RateLimitDecision) -> dict[str, str]:
    return {
        "RateLimit-Limit": str(decision.limit),
        "RateLimit-Remaining": str(decision.remaining),
    }


def _rate_limited_response(decision: RateLimitDecision) -> JSONResponse:
    headers = _rate_limit_headers(decision)
    headers.update({
        "Cache-Control": "no-store",
        "Retry-After": str(decision.retry_after_seconds),
    })
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers=headers,
        content={
            "error": {
                "code": "rate_limited",
                "message": "Too many requests. Try again later.",
            }
        },
    )


def _query_response(
    answer: GeneratedAnswer,
    *,
    user_id: str,
    session_id: UUID,
) -> QueryResponse:
    return QueryResponse(
        answer=answer.text,
        abstained=not answer.citations,
        user_id=user_id,
        session_id=str(session_id),
        citations=[
            CitationResponse(
                number=citation.number,
                record_id=citation.record_id,
                source=citation.source,
                page_number=citation.page_number,
                chunk_index=citation.chunk_index,
            )
            for citation in answer.citations
        ],
    )


def create_app(
    *,
    pipeline: AnswerPipeline | None = None,
    pipeline_factory: PipelineFactory = build_default_pipeline,
    authenticator: Authenticator | None = None,
    authenticator_factory: AuthenticatorFactory = build_default_authenticator,
    document_permissions: PermissionResolver | None = None,
    document_permissions_factory: PermissionResolverFactory = (
        build_default_document_permissions
    ),
    rate_limiter: RateLimiter | None = None,
    rate_limiter_factory: RateLimiterFactory = build_default_rate_limiter,
    request_logger: logging.Logger = LOGGER,
) -> FastAPI:
    """Build an application with an injectable pipeline for deterministic tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.pipeline = pipeline
        application.state.authenticator = authenticator
        application.state.document_permissions = document_permissions
        application.state.rate_limiter = rate_limiter
        if application.state.pipeline is None:
            try:
                application.state.pipeline = await run_in_threadpool(
                    pipeline_factory
                )
            except Exception as error:
                application.state.pipeline = None
                request_logger.error(
                    "RAG pipeline initialization failed (type=%s)",
                    type(error).__name__,
                )
        if application.state.authenticator is None:
            try:
                application.state.authenticator = await run_in_threadpool(
                    authenticator_factory
                )
            except Exception as error:
                application.state.authenticator = None
                request_logger.error(
                    "API authentication initialization failed (type=%s)",
                    type(error).__name__,
                )
        if application.state.document_permissions is None:
            try:
                application.state.document_permissions = await run_in_threadpool(
                    document_permissions_factory
                )
            except Exception as error:
                application.state.document_permissions = None
                request_logger.error(
                    "Document permission initialization failed (type=%s)",
                    type(error).__name__,
                )
        if application.state.rate_limiter is None:
            try:
                application.state.rate_limiter = await run_in_threadpool(
                    rate_limiter_factory
                )
            except Exception as error:
                application.state.rate_limiter = None
                request_logger.error(
                    "Rate limiter initialization failed (type=%s)",
                    type(error).__name__,
                )
        yield

    application = FastAPI(
        title="RAG System API",
        version="8.0.0",
        lifespan=lifespan,
        redoc_url=None,
    )

    @application.middleware("http")
    async def correlate_and_log_request(
        request: Request,
        call_next: Callable,
    ) -> Response:
        request_id = normalize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started_at = time.monotonic()
        event_fields = {
            "method": request.method,
            "path": request.url.path,
        }

        with bind_log_context(request_id=request_id):
            request_logger.info("request.started", extra=event_fields)
            try:
                result = await call_next(request)
            except Exception as error:
                request_logger.error(
                    "request.failed",
                    extra={
                        **event_fields,
                        "exception_type": type(error).__name__,
                    },
                )
                result = JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    headers={"Cache-Control": "no-store"},
                    content={
                        "error": {
                            "code": "internal_error",
                            "message": "The service encountered an unexpected error.",
                        }
                    },
                )

            result.headers["X-Request-ID"] = request_id
            request_logger.info(
                "request.completed",
                extra={
                    **event_fields,
                    "status_code": result.status_code,
                    "duration_ms": (time.monotonic() - started_at) * 1_000,
                    "user_id": getattr(request.state, "user_id", None),
                    "session_id": getattr(request.state, "session_id", None),
                },
            )
            return result

    @application.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
    )
    async def liveness() -> HealthResponse:
        return HealthResponse(status="alive")

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
        tags=["health"],
    )
    async def readiness(request: Request) -> HealthResponse | JSONResponse:
        if (
            request.app.state.pipeline is None
            or request.app.state.authenticator is None
            or request.app.state.document_permissions is None
            or request.app.state.rate_limiter is None
        ):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready"},
            )
        return HealthResponse(status="ready")

    bearer_scheme = HTTPBearer(auto_error=False)

    @application.post(
        "/v1/query",
        response_model=QueryResponse,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
            status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        },
        tags=["rag"],
    )
    async def query(
        payload: QueryRequest,
        request: Request,
        response: Response,
        credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
        requested_session_id: UUID | None = Header(
            default=None,
            alias="X-Session-ID",
        ),
    ) -> QueryResponse | JSONResponse:
        resolved_pipeline = request.app.state.pipeline
        resolved_authenticator = request.app.state.authenticator
        resolved_permissions = request.app.state.document_permissions
        resolved_rate_limiter = request.app.state.rate_limiter
        if (
            resolved_pipeline is None
            or resolved_authenticator is None
            or resolved_permissions is None
            or resolved_rate_limiter is None
        ):
            return _safe_unavailable_response()

        user_id = None
        if credentials is not None:
            user_id = resolved_authenticator.authenticate(credentials.credentials)
        if user_id is None:
            return _unauthorized_response()
        request.state.user_id = user_id
        resolved_session_id = requested_session_id or uuid4()
        request.state.session_id = str(resolved_session_id)

        allowed_sources = resolved_permissions.allowed_sources(user_id)
        if allowed_sources is None:
            return _forbidden_response()

        try:
            rate_limit_decision = resolved_rate_limiter.check(user_id)
        except Exception as error:
            request_logger.error(
                "Rate limiter check failed (type=%s)",
                type(error).__name__,
                extra={
                    "user_id": user_id,
                    "session_id": str(resolved_session_id),
                    "exception_type": type(error).__name__,
                },
            )
            return _safe_unavailable_response()
        if not rate_limit_decision.allowed:
            return _rate_limited_response(rate_limit_decision)

        try:
            answer = await run_in_threadpool(
                resolved_pipeline.ask,
                payload.question,
                top_k=payload.top_k,
                allowed_sources=allowed_sources,
            )
        except Exception as error:
            request_logger.error(
                "RAG query execution failed (type=%s)",
                type(error).__name__,
                extra={
                    "user_id": user_id,
                    "session_id": str(resolved_session_id),
                    "exception_type": type(error).__name__,
                },
            )
            return _safe_unavailable_response()
        for header, value in _rate_limit_headers(rate_limit_decision).items():
            response.headers[header] = value
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Session-ID"] = str(resolved_session_id)
        return _query_response(
            answer,
            user_id=user_id,
            session_id=resolved_session_id,
        )

    return application


app = create_app()
