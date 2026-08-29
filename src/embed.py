"""Create chunk embeddings and persist them in a Chroma collection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, TypeVar

from src.chunk import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Tokenizer,
    chunk_text,
)
from src.ingest import RawDocument


DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_CHROMA_PATH = Path("data/chroma")

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]
Token = TypeVar("Token")


class EmbeddingProvider(Protocol):
    """Embedding operations needed by indexing and retrieval."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, records: Sequence[EmbeddedChunk]) -> None: ...

    def delete(self, record_ids: Sequence[str]) -> None: ...

    def all_chunks(self) -> list[StoredChunk]: ...


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A source chunk with its vector, stable ID, and citation metadata."""

    id: str
    text: str
    embedding: list[float]
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A persisted chunk loaded without its embedding vector."""

    id: str
    text: str
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A nearest-neighbor result ordered by increasing vector distance."""

    id: str
    text: str
    metadata: Metadata
    distance: float
    rank: int


class SentenceTransformerEmbeddingProvider:
    """Local document/query embeddings backed by Sentence Transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as error:
                if error.name != "sentence_transformers":
                    raise RuntimeError(
                        "sentence-transformers could not load one of its installed "
                        "dependencies. Use a clean virtual environment and reinstall "
                        "requirements.txt."
                    ) from error
                raise RuntimeError(
                    "Local embeddings require sentence-transformers. Install "
                    "dependencies with `python3 -m pip install -r requirements.txt`."
                ) from error
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is installed but could not be imported. "
                    "Use a clean virtual environment and reinstall requirements.txt."
                ) from error
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _encoding_options() -> dict[str, bool]:
        return {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }

    @staticmethod
    def _as_document_vectors(value: Any) -> list[list[float]]:
        raw_vectors = value.tolist() if hasattr(value, "tolist") else value
        return [[float(item) for item in vector] for vector in raw_vectors]

    @staticmethod
    def _as_query_vector(value: Any) -> list[float]:
        raw_vector = value.tolist() if hasattr(value, "tolist") else value
        return [float(item) for item in raw_vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode_document(
            list(texts),
            **self._encoding_options(),
        )
        return self._as_document_vectors(vectors)

    def embed_query(self, text: str) -> list[float]:
        vector = self._get_model().encode_query(
            text,
            **self._encoding_options(),
        )
        return self._as_query_vector(vector)


class ChromaVectorStore:
    """Persistent Chroma storage for precomputed embedding records."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import chromadb
            except ModuleNotFoundError as error:
                if error.name != "chromadb":
                    raise RuntimeError(
                        "chromadb could not load one of its installed dependencies. "
                        "Use a clean virtual environment and reinstall requirements.txt."
                    ) from error
                raise RuntimeError(
                    "Vector persistence requires chromadb. Install dependencies "
                    "with `python3 -m pip install -r requirements.txt`."
                ) from error
            except ImportError as error:
                raise RuntimeError(
                    "chromadb is installed but could not be imported. Use a clean "
                    "virtual environment and reinstall requirements.txt."
                ) from error
            client = chromadb.PersistentClient(path=str(path))

        self.collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    def upsert(self, records: Sequence[EmbeddedChunk]) -> None:
        if not records:
            return
        self.collection.upsert(
            ids=[record.id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
        )

    def delete(self, record_ids: Sequence[str]) -> None:
        ids = list(record_ids)
        if ids:
            self.collection.delete(ids=ids)

    def all_chunks(self) -> list[StoredChunk]:
        """Load the persisted text corpus for non-vector retrieval strategies."""

        response = self.collection.get(include=["documents", "metadatas"])
        try:
            ids = response.get("ids") or []
            documents = response.get("documents") or []
            metadatas = response.get("metadatas") or []
        except (AttributeError, TypeError) as error:
            raise ValueError("Chroma returned a malformed corpus response") from error

        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("Chroma returned malformed corpus columns")

        chunks: list[StoredChunk] = []
        for record_id, text, metadata in zip(ids, documents, metadatas):
            if not isinstance(record_id, str) or not isinstance(text, str):
                raise ValueError("Chroma returned malformed corpus data")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("Chroma returned malformed corpus metadata")
            chunks.append(
                StoredChunk(
                    id=record_id,
                    text=text,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest stored chunks for one precomputed query vector."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")
        if allowed_sources is not None and not allowed_sources:
            return []

        query_options: dict[str, Any] = dict(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if allowed_sources is not None:
            query_options["where"] = {
                "source": {"$in": sorted(allowed_sources)},
            }
        response = self.collection.query(**query_options)

        try:
            id_batches = response.get("ids") or [[]]
            document_batches = response.get("documents") or [[]]
            metadata_batches = response.get("metadatas") or [[]]
            distance_batches = response.get("distances") or [[]]
            ids = id_batches[0]
            documents = document_batches[0]
            metadatas = metadata_batches[0]
            distances = distance_batches[0]
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError("Chroma returned a malformed query response") from error

        if not ids:
            return []
        if not (
            len(ids) == len(documents) == len(metadatas) == len(distances)
        ):
            raise ValueError("Chroma returned malformed result columns")

        results: list[RetrievedChunk] = []
        for rank, (record_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            if not isinstance(record_id, str) or not isinstance(text, str):
                raise ValueError("Chroma returned malformed document data")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("Chroma returned malformed metadata")
            if distance is None:
                raise ValueError("Chroma returned a result without a distance")
            results.append(
                RetrievedChunk(
                    id=record_id,
                    text=text,
                    metadata=dict(metadata or {}),
                    distance=float(distance),
                    rank=rank,
                )
            )
        return results


def _record_id(
    document: RawDocument,
    *,
    chunk_index: int,
    token_start: int,
    token_end: int,
    text: str,
) -> str:
    identity = "\0".join(
        (
            document.source,
            str(document.page_number or ""),
            str(chunk_index),
            str(token_start),
            str(token_end),
            text,
        )
    )
    return sha256(identity.encode("utf-8")).hexdigest()


def index_documents(
    documents: Sequence[RawDocument],
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    tokenizer: Tokenizer[Token] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[EmbeddedChunk]:
    """Chunk raw documents, embed the chunks once, and atomically upsert them."""

    pending: list[tuple[str, str, Metadata]] = []
    for document in documents:
        chunks = chunk_text(
            document.text,
            chunk_size=chunk_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
        for chunk in chunks:
            metadata: Metadata = {
                "source": document.source,
                "document_type": document.document_type,
                "chunk_index": chunk.index,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
                "token_count": chunk.token_count,
            }
            if document.page_number is not None:
                metadata["page_number"] = document.page_number

            pending.append(
                (
                    _record_id(
                        document,
                        chunk_index=chunk.index,
                        token_start=chunk.token_start,
                        token_end=chunk.token_end,
                        text=chunk.text,
                    ),
                    chunk.text,
                    metadata,
                )
            )

    if not pending:
        return []

    embeddings = embedding_provider.embed_documents([item[1] for item in pending])
    if len(embeddings) != len(pending):
        raise ValueError(
            "The embedding provider returned a different number of embeddings "
            f"({len(embeddings)}) than input chunks ({len(pending)})."
        )

    records = [
        EmbeddedChunk(
            id=record_id,
            text=text,
            embedding=[float(value) for value in embedding],
            metadata=metadata,
        )
        for (record_id, text, metadata), embedding in zip(pending, embeddings)
    ]
    vector_store.upsert(records)
    return records
