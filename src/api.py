"""Typed HTTP boundary for the existing deterministic RAG pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from scripts.ask import build_phase2_retriever
from src.embed import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
)
from src.generate import GeneratedAnswer
from src.llm import DEFAULT_GROQ_MODEL, GroqLanguageModel
from src.pipeline import RAGPipeline
from src.rerank import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
)
from src.retrieve import DEFAULT_TOP_K


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_QUESTION_LENGTH = 4_096
MAX_TOP_K = 20


class AnswerPipeline(Protocol):
    """The small application interface required by the HTTP layer."""

    def ask(self, question: str, *, top_k: int) -> GeneratedAnswer: ...


PipelineFactory = Callable[[], AnswerPipeline]


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


def _safe_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": {
                "code": "rag_unavailable",
                "message": "The RAG service is temporarily unavailable.",
            }
        },
    )


def _query_response(answer: GeneratedAnswer) -> QueryResponse:
    return QueryResponse(
        answer=answer.text,
        abstained=not answer.citations,
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
) -> FastAPI:
    """Build an application with an injectable pipeline for deterministic tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.pipeline = pipeline
        if application.state.pipeline is None:
            try:
                application.state.pipeline = await run_in_threadpool(
                    pipeline_factory
                )
            except Exception as error:
                application.state.pipeline = None
                LOGGER.error(
                    "RAG pipeline initialization failed (type=%s)",
                    type(error).__name__,
                )
        yield

    application = FastAPI(
        title="RAG System API",
        version="4.0.0",
        lifespan=lifespan,
        redoc_url=None,
    )

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
        if request.app.state.pipeline is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready"},
            )
        return HealthResponse(status="ready")

    @application.post(
        "/v1/query",
        response_model=QueryResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
        tags=["rag"],
    )
    async def query(
        payload: QueryRequest,
        request: Request,
    ) -> QueryResponse | JSONResponse:
        resolved_pipeline = request.app.state.pipeline
        if resolved_pipeline is None:
            return _safe_unavailable_response()
        try:
            answer = await run_in_threadpool(
                resolved_pipeline.ask,
                payload.question,
                top_k=payload.top_k,
            )
        except Exception as error:
            LOGGER.error(
                "RAG query execution failed (type=%s)",
                type(error).__name__,
            )
            return _safe_unavailable_response()
        return _query_response(answer)

    return application


app = create_app()
