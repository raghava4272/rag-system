# RAG System — Phase 1: Fundamentals

## Overview

This is Phase 1 of a three-phase build toward a production-grade Retrieval
Augmented Generation (RAG) system — a domain-specific "ask my docs" tool that
answers questions over a chosen corpus with accurate citations.

**Phase 1 goal:** get a working end-to-end pipeline — ingest documents, chunk
them, embed them, store them in a vector database, retrieve the top-k
relevant chunks for a query, and generate a cited answer. No hybrid search,
no re-ranking, no CI/eval loop yet — those are Phase 2 and Phase 3.

## Corpus

Pick one domain to focus the system on. Examples: technical documentation,
research papers, legal contracts, healthcare documents. Put source files in
`data/raw/`. (Default assumption for this scaffold: a folder of technical
docs / markdown + PDFs — swap in your own corpus.)

## Scope — What Phase 1 Delivers

- [ ] Document ingestion for PDF, Markdown, and web page sources (PDF and
      Markdown complete; web pages remain optional)
- [x] Chunking: ~500–800 tokens per chunk, 100-token overlap
- [x] Embedding generation for each chunk
- [x] Vector store persistence with Chroma
- [x] Basic retrieval: top-k semantic search over the vector store
- [x] Answer generation that cites the source chunk(s) used
- [x] A simple CLI or script entry point to ask a question and get an
      answer + citations

**Out of scope for Phase 1** (comes later): BM25 hybrid search, cross-encoder
re-ranking, hallucination-guard/decline-to-answer logic, golden eval dataset,
CI wiring.

## Corpus Sourcing

Recommended source for Phase 1: **LangChain's official documentation**, since
it's clean markdown, actively maintained, and directly relevant to this
project's domain. Mix in a couple of PDFs so both loaders get exercised.

**Markdown (primary):**

- Source repo: https://github.com/langchain-ai/docs
- Clone it and pull `.md` / `.mdx` files from a coherent subfolder (e.g.
  `src/oss/`) rather than the whole repo — aim for ~20-30 pages.
  ```bash
  git clone https://github.com/langchain-ai/docs.git /tmp/langchain-docs
  cp /tmp/langchain-docs/src/oss/**/*.md* data/raw/
  ```
- The site also publishes a full page index at `docs.langchain.com/llms.txt`
  if you want to browse before picking pages.

**PDFs (secondary, for loader/chunking variety):**

- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et
  al.) — https://arxiv.org/pdf/2005.11401
- ReAct: Synergizing Reasoning and Acting in Language Models —
  https://arxiv.org/pdf/2210.03629
- Add more from arXiv (`cs.CL` / `cs.LG` categories) as needed — each paper
  is available directly at `arxiv.org/pdf/<id>`.

Target for Phase 1: ~20-40 total documents, mostly markdown, a handful of
PDFs — enough to meaningfully test citations without slowing iteration.

## Tech Stack

| Component       | Choice                                      |
| --------------- | ------------------------------------------- |
| Orchestration   | Small provider-neutral Python modules       |
| Vector Store    | ChromaDB with local persistence             |
| Embeddings      | `Alibaba-NLP/gte-modernbert-base` locally   |
| LLM (answering) | Groq API with `openai/gpt-oss-20b` default  |

## Project Structure

```
rag-system/
├── data/
│   ├── raw/              # source documents (pdf, md, html)
│   └── processed/        # chunked + metadata, if cached to disk
├── src/
│   ├── ingest.py         # load PDFs/markdown/web pages -> raw text
│   ├── chunk.py          # split text into 500-800 token chunks, 100 overlap
│   ├── embed.py          # generate + store embeddings in vector store
│   ├── retrieve.py       # top-k similarity search
│   ├── generate.py       # build prompt with retrieved chunks, call LLM, cite sources
│   ├── llm.py            # Groq language-model adapter
│   └── pipeline.py        # wires ingest -> chunk -> embed (index build)
├── scripts/
│   ├── build_index.py    # one-off script: run full ingestion pipeline
│   └── ask.py             # CLI: ask a question, print answer + citations
├── tests/
│   └── test_chunking.py  # sanity checks on chunk sizes/overlap
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```
2. Create a free Groq API key at https://console.groq.com/keys, then configure
   it locally:
   ```bash
   cp .env.example .env
   # Edit .env and set GROQ_API_KEY. Never commit the real key.
   ```
3. Drop source documents into `data/raw/`.

## Usage

Build the vector index:

```bash
python3 scripts/build_index.py
```

Ask a question:

```bash
python3 scripts/ask.py "What does the document say about X?"
```

Expected output: an answer plus a list of source chunks/document names it
was drawn from.

## Definition of Done (Phase 1)

- Running `build_index.py` ingests all files in `data/raw/`, chunks them
  correctly (500–800 tokens, 100-token overlap — verify with a unit test),
  and persists embeddings to the vector store.
- Running `ask.py "<question>"` retrieves the top-k relevant chunks and
  returns an answer with citations pointing to the specific source
  document/section each claim came from.
- Pipeline works on at least PDF and Markdown inputs.

## Notes for Building with Codex

Suggested build order for an agentic coding session:

1. Scaffold the project structure and `requirements.txt`.
2. Implement `ingest.py` for PDF + Markdown (web pages can come later).
3. Implement `chunk.py` with a token-based splitter (verify sizes with a
   test) — do this before embedding so chunk quality is confirmed early.
4. Implement `embed.py` to write chunks + embeddings + metadata (source
   file, chunk index) into Chroma.
5. Implement `retrieve.py` for top-k cosine/similarity search.
6. Implement `generate.py` to construct a grounded prompt and return an
   answer with citations back to source metadata.
7. Wire it together in `pipeline.py` / `scripts/build_index.py` and
   `scripts/ask.py`.
8. Add a couple of tests for chunk size/overlap and for citation presence
   in the output.

## Next Steps (Preview)

- **Phase 2:** hybrid retrieval (BM25 + vector), cross-encoder re-ranking,
  and enforced citation/decline-to-answer logic.
- **Phase 3:** golden eval dataset (50–200 QA pairs), offline faithfulness
  evaluation (Ragas), and CI pipeline gating on quality thresholds.
