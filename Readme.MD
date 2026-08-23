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


## Phase 2: Production Quality

### Overview

Phase 2 upgrades the Phase 1 pipeline from "works on a demo" to "holds up
under real queries." It combines keyword and semantic search, re-ranks the
resulting candidates, and explicitly refuses to answer when the evidence is
too weak.

**Phase 2 goal:** replace naive top-k vector search with hybrid retrieval and
re-ranking, then enforce traceable citation coverage — no supported context,
no answer.

### Scope — What Phase 2 Delivers

- [x] BM25 keyword search alongside vector search
- [x] Reciprocal Rank Fusion (RRF) of BM25 and vector rankings
- [x] Local cross-encoder candidate re-ranking
- [x] Configurable relevance threshold and decline-to-answer path
- [x] Citation validation for every answer paragraph and list item
- [x] Phase 1 regression checks and unsupported-query checks

**Out of scope for Phase 2:** a large golden evaluation dataset, automated
faithfulness scoring, and CI quality gates. Those belong to Phase 3.

### Tech Stack Additions

| Component      | Choice                                         |
| -------------- | ---------------------------------------------- |
| Keyword search | `rank-bm25` / Okapi BM25                       |
| Fusion         | Reciprocal Rank Fusion with `k=60`             |
| Re-ranking     | `cross-encoder/ms-marco-MiniLM-L6-v2` locally  |
| Safety gate    | Sigmoid relevance threshold (`0.5` by default) |

### Project Structure Additions

```text
rag-system/
├── data/phase2_regression.json
├── src/
│   ├── retrieve_bm25.py
│   ├── hybrid_retrieve.py
│   ├── rerank.py
│   ├── regression.py
│   └── generate.py
├── scripts/check_phase2.py
└── tests/
    ├── test_bm25_retrieval.py
    ├── test_hybrid_retrieval.py
    ├── test_rerank.py
    ├── test_guarded_retrieval.py
    ├── test_citation_guard.py
    ├── test_phase2_wiring.py
    └── test_phase2_regression.py
```

### Usage

The original command now uses the complete Phase 2 retrieval path:

```bash
python scripts/ask.py "What does the document say about X?"
```

Optional tuning controls:

```bash
python scripts/ask.py "What does the document say about X?" \
  --candidate-k 10 \
  --top-k 5 \
  --min-relevance-score 0.5
```

Run the local retrieval regression gate without calling Groq:

```bash
python scripts/check_phase2.py
```

### Regression Result

Real-corpus check on 2026-08-22:

| Metric                     | Phase 1 vector | Phase 2 |
| -------------------------- | -------------- | ------- |
| Hit@1                      | 80%            | 80%     |
| Hit@3                      | 80%            | 100%    |
| Unsupported decline rate   | —              | 100%    |
| Supported-query regression | —              | None    |

Phase 2 recovered the expected agentic-RAG source at rank 3 after vector-only
retrieval missed it. All 84 automated tests pass.

### Definition of Done (Phase 2)

- [x] Hybrid retrieval returns fused BM25 and vector candidates.
- [x] The cross-encoder improves measured top-3 relevance without regressions.
- [x] Unsupported questions return an explicit decline without calling Groq.
- [x] Every factual paragraph or list item contains a valid source citation.
- [x] Phase 1 example queries still return correct, cited answers.

Phase 3: CI/CD Integration
Overview

Phase 3 turns the pipeline into something you can trust to not silently regress. Instead of eyeballing a few example queries, you build a fixed "golden" test set, score every answer automatically for faithfulness, and fail the build if quality drops — the same way you'd gate on unit tests.

Phase 3 goal: a golden QA dataset, an offline evaluation script that scores faithfulness (and ideally answer relevance/context precision), and a CI job that runs it on every pull request and fails below a threshold.

Scope — What Phase 3 Delivers
 A golden evaluation dataset of 50–200 manually verified question/expected-answer/expected-source triples
 An offline evaluation script that runs the full pipeline against the golden set and scores faithfulness (and other relevant metrics) using Ragas
 A defined quality threshold (e.g. minimum average faithfulness score)
 A CI workflow that runs the eval script on every pull request
 CI fails the build automatically if scores drop below threshold
 A human-readable eval report (pass/fail per question, aggregate score) generated as a CI artifact

Out of scope for Phase 3: anything beyond automated regression gating — this phase is about locking in the quality Phase 1 and 2 already built, not adding new retrieval/generation features.

Tech Stack Additions
Component	Choice
Evaluation framework	Ragas (faithfulness, answer relevancy, context precision/recall)
CI platform	GitHub Actions (or your existing CI provider)
Dataset format	CSV or JSONL: question, ground_truth_answer, ground_truth_sources
Project Structure Additions
rag-system/
├── eval/
│   ├── golden_dataset.jsonl   # 50-200 manually verified QA pairs
│   ├── run_eval.py            # runs pipeline against golden set, scores with Ragas
│   └── report.py              # formats results into a readable report (CI artifact)
├── .github/
│   └── workflows/
│       └── rag-eval.yml       # CI workflow: run_eval.py on every PR, fail below threshold
Building the Golden Dataset
Pull real or realistic questions covering your corpus's breadth — include easy factual lookups, multi-chunk synthesis questions, and a few questions the corpus deliberately can't answer (to test the decline-to-answer path from Phase 2).
For each question, manually verify: the expected answer, and which source document(s)/chunk(s) should be cited.
Start smaller (e.g. 50) and grow toward 200 as you find failure modes worth locking in as regression tests.

Example record:

json
{
  "question": "What retriever does LangChain's hybrid search combine with vector search?",
  "ground_truth_answer": "BM25 keyword search.",
  "ground_truth_sources": ["docs/retrieval/hybrid-search.md"]
}
Usage (Phase 3 additions)

Run the evaluation locally:

bash
python eval/run_eval.py --dataset eval/golden_dataset.jsonl --threshold 0.8

Expected output: per-question scores, an aggregate faithfulness score, and a non-zero exit code if the aggregate falls below --threshold — this exit code is what CI checks.

Definition of Done (Phase 3)
golden_dataset.jsonl contains at least 50 manually verified QA pairs covering a representative range of the corpus.
run_eval.py runs the full Phase 1+2 pipeline against every golden question and computes Ragas faithfulness (plus any additional metrics you choose) per question and in aggregate.
A quality threshold is defined and enforced — the script exits non-zero when the aggregate score falls below it.
.github/workflows/rag-eval.yml runs run_eval.py on every pull request and the PR check fails automatically when the script exits non-zero.
A report artifact (e.g. markdown or HTML summary) is generated so a reviewer can see which questions failed and why without re-running the eval locally.
Notes for Building with Codex

Suggested build order:

Draft golden_dataset.jsonl with an initial ~50 questions — do this manually/carefully, since the whole phase's value depends on this data being genuinely correct, not generated and rubber-stamped.
Implement run_eval.py: load the dataset, run each question through the Phase 1+2 pipeline (ask.py logic, called as a function rather than a CLI), collect the answer + retrieved contexts, and pass them to Ragas's faithfulness metric.
Add threshold logic: compute the aggregate score, compare to --threshold, exit non-zero on failure.
Implement report.py to turn per-question results into a readable summary (which questions passed/failed, scores, and the actual vs. expected answer for failures).
Write .github/workflows/rag-eval.yml: trigger on pull_request, install dependencies, run run_eval.py, upload the report as a CI artifact, and let the script's exit code determine pass/fail.
Test the gate deliberately: intentionally break something (e.g. lower top_k to 1) and confirm CI actually fails, then revert.
Project Complete

With Phase 3 done, the system ingests documents, retrieves with hybrid search and re-ranking, answers with enforced citations (or explicitly declines), and is protected by an automated regression gate in CI — a production-grade RAG pipeline end to end.

## Docker Compose Workflow

The application runs as one-shot container jobs; it does not need a web UI or
long-running server. Build the runtime image once:

```bash
docker compose build
```

Build or update the index from the host's read-only `data/raw/` directory:

```bash
docker compose run --rm index
```

Ask a question using the same persistent index:

```bash
docker compose run --rm ask "What is reciprocal rank fusion?"
```

Run the Phase 3 quality gate and print its persisted report:

```bash
docker compose run --rm evaluate
docker compose run --rm report
```

Compose reads `GROQ_API_KEY` and `GROQ_MODEL` from the local `.env` file and
injects them only into the jobs that call Groq. It never adds the key to the
image. Named volumes persist the Chroma index, downloaded model cache, and
evaluation report across disposable containers.

The Dockerfile selects `requirements-linux-amd64.lock` or
`requirements-linux-arm64.lock` from BuildKit's target architecture. Compose
does not force a platform, so it builds and runs natively on both Intel/AMD
Linux and ARM64 machines such as Apple Silicon.

To inspect the persistent resources:

```bash
docker volume ls --filter label=com.docker.compose.project=rag-system
```

Deleting the `rag-system_rag-chroma` volume deletes the built index, so do not
remove Compose volumes unless you intend to rebuild it.
