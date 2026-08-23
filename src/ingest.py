"""Load supported local source files into citation-friendly raw documents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


SUPPORTED_SUFFIXES = frozenset({".md", ".mdx", ".pdf"})


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Extracted source text and the metadata needed for later citations."""

    text: str
    source: str
    document_type: str
    page_number: int | None = None


class PdfPage(Protocol):
    def extract_text(self) -> str | None: ...


class PdfReader(Protocol):
    pages: Sequence[PdfPage]


PdfReaderFactory = Callable[[Path], PdfReader]


def _open_pdf_reader(path: Path) -> PdfReader:
    try:
        from pypdf import PdfReader as PyPdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF ingestion requires pypdf. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from error

    return PyPdfReader(path)


def load_document(
    source: str | Path,
    *,
    pdf_reader_factory: PdfReaderFactory | None = None,
) -> list[RawDocument]:
    """Load one Markdown/MDX file or one PDF into raw documents.

    Markdown files yield one document. PDFs yield one document per nonempty page
    so citations can retain their original page numbers.
    """

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)

    suffix = path.suffix.casefold()
    if suffix in {".md", ".mdx"}:
        return [
            RawDocument(
                text=path.read_text(encoding="utf-8"),
                source=str(path),
                document_type="markdown",
            )
        ]
    if suffix == ".pdf":
        reader_factory = pdf_reader_factory or _open_pdf_reader
        reader = reader_factory(path)
        documents: list[RawDocument] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            documents.append(
                RawDocument(
                    text=text.strip(),
                    source=str(path),
                    document_type="pdf",
                    page_number=page_number,
                )
            )
        return documents

    raise ValueError(
        f"Unsupported document type {path.suffix!r} for {path}; "
        f"expected one of {sorted(SUPPORTED_SUFFIXES)}"
    )


def discover_source_files(source_directory: str | Path) -> list[Path]:
    """Return supported source files recursively in stable relative-path order."""
    directory = Path(source_directory)
    if not directory.exists():
        raise FileNotFoundError(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.relative_to(directory).as_posix().casefold(),
    )


def load_documents(source_directory: str | Path) -> list[RawDocument]:
    """Recursively load supported files from a directory in stable order."""

    documents: list[RawDocument] = []
    for path in discover_source_files(source_directory):
        documents.extend(load_document(path))
    return documents
