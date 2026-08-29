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

## Phase 4: API Foundation

Phase 4 puts a typed HTTP boundary in front of the existing deterministic RAG
pipeline. It does not make retrieval agentic yet. The API gives later phases a
stable place to add authentication, sessions, permissions, streaming, rate
limits, and finally agent decisions without coupling those concerns to the CLI.

### Scope

- [x] `POST /v1/query` with validated question and `top_k` fields
- [x] Structured answer, abstention flag, and citation objects
- [x] Separate liveness and readiness endpoints
- [x] One pipeline initialization per server lifetime
- [x] Safe `503` responses that do not expose provider or configuration details
- [x] Interactive OpenAPI documentation at `/docs`
- [x] A non-root, read-only, health-checked Docker Compose API service
- [x] Exact AMD64 and ARM64 dependency locks for FastAPI and Uvicorn

Authentication and agent tool selection were deliberately not part of this
phase. Authentication is added in Phase 5; document authorization and agent
tool selection still come later.

### Run the API with Docker

Build the index before starting the server:

```bash
docker compose build
docker compose run --rm index
docker compose up api
```

Check whether the process is alive and whether its RAG dependencies initialized:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

Ask a question:

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is reciprocal rank fusion?","top_k":5}'
```

Open `http://localhost:8000/docs` to try the same contract interactively. If
the Groq key or another startup dependency is unavailable, liveness remains
healthy while readiness and query requests return `503`.

## Phase 5: Authentication and Request Identity

Phase 5 establishes who is making each API request before document permissions
or agent decisions are introduced. The server never trusts a client-provided
user ID. Instead, each bearer key hash is mapped to a user ID in runtime
configuration, and the presented key is verified with a constant-time hash
comparison.

### Scope

- [x] Bearer authentication on `POST /v1/query`
- [x] Raw API keys kept out of source code, images, and server configuration
- [x] SHA-256 key hashes mapped to trusted user IDs at runtime
- [x] Generic `401` response for both missing and invalid credentials
- [x] Validated client UUID sessions or server-generated UUIDv4 sessions
- [x] User and session identity returned in the structured response
- [x] `Cache-Control: no-store` on authenticated query responses
- [x] Public liveness/readiness probes for container orchestration
- [x] Readiness failure when authentication is not configured

The session ID is currently a correlation identity, not stored conversation
memory. Document-level authorization is added in Phase 6.

### Create a Local API Key

Generate a strong key for one user:

```bash
python3 scripts/create_api_key.py --user-id local-user
```

The command displays the raw key once and prints a
`RAG_API_KEY_HASHES={...}` line for `.env`. Save the raw key in a password or
secret manager and add only the printed hash mapping to `.env`. To configure
multiple users, combine their user-to-hash entries into one JSON object.

Provide the saved raw key to your API client through a secret environment
variable named `RAG_API_KEY`; it is not the same value as
`RAG_API_KEY_HASHES`. The query example above reads it from that client-side
variable. An optional `X-Session-ID` header may contain an existing UUID; when
omitted, the server generates one and returns it in both the JSON response and
the `X-Session-ID` response header.

## Phase 6: Document Authorization

Authentication proves who sent a query. Authorization now determines which
indexed sources that identity may retrieve. The API maps the authenticated
server-side user ID to an exact set of source identifiers and passes that scope
through semantic retrieval, BM25 retrieval, hybrid fusion, and reranking.

### Scope

- [x] Deny access when an authenticated user has no permission rule
- [x] Exact source matching with no wildcard or prefix expansion
- [x] Vector filtering inside the Chroma query
- [x] BM25 filtering before result ordering and rank assignment
- [x] One permission scope shared by hybrid retrieval and reranking
- [x] Final pipeline guard before retrieved content reaches Groq
- [x] Empty permission sets safely produce no retrieved context
- [x] Generic `403` responses that do not reveal source names
- [x] Readiness failure when permission configuration is missing or invalid

### Configure Local Document Access

Add a `RAG_DOCUMENT_PERMISSIONS` JSON object to `.env`. Its user IDs must match
the IDs in `RAG_API_KEY_HASHES`, and every source string must exactly match the
`source` metadata stored during indexing. With the documented `data/raw` index
command, the two included papers can be granted to `local-user` like this:

```dotenv
RAG_DOCUMENT_PERMISSIONS={"local-user":["data/raw/papers/rag-lewis-2020.pdf","data/raw/papers/react-yao-2022.pdf"]}
```

An empty list is a valid explicit rule and gives that user access to no
documents. A missing user rule returns `403`. Wildcards are intentionally not
supported because exact rules are easier to audit and cannot silently expand
when a new document is indexed.

Permission checks happen before evidence is sent to the language model. Chroma
receives an exact metadata filter, BM25 removes out-of-scope candidates before
ranking, and the pipeline independently rejects any out-of-scope result as a
last line of defense.

## Phase 7: Per-User Rate Limiting

Phase 7 protects the expensive retrieval and Groq path from sustained request
bursts. Each authenticated server-side user ID receives an independent token
bucket. Session IDs and client-provided `X-User-ID` headers do not affect the
bucket, so opening a new session or spoofing a header cannot reset the quota.

### Scope

- [x] Thread-safe token consumption for concurrent API requests
- [x] Independent limits keyed by trusted authenticated user ID
- [x] Smooth token refill instead of fixed-window boundary bursts
- [x] `429 Too Many Requests` before retrieval or Groq execution
- [x] `RateLimit-Limit` and `RateLimit-Remaining` response headers
- [x] `Retry-After` on blocked requests
- [x] Safe defaults with bounded runtime configuration
- [x] Stale identity cleanup to prevent unbounded bucket growth
- [x] Readiness failure if limiter initialization is invalid
- [x] Public health probes remain outside the user quota

The defaults allow 30 requests per 60 seconds, with an initial burst capacity
of 30. Override them in `.env` when needed:

```dotenv
RAG_RATE_LIMIT_REQUESTS=30
RAG_RATE_LIMIT_WINDOW_SECONDS=60
```

The values must be integers. Requests are bounded to `1-10000`, and the window
is bounded to `1-86400` seconds. Invalid configuration makes readiness fail
instead of silently disabling the limiter.

This limiter intentionally matches the current Compose deployment: one Uvicorn
process in one API container. Its counters are held in that process. Before
running multiple workers or replicas, replace the limiter backend with a shared
atomic store such as Redis or enforce the same authenticated-user policy at an
API gateway. An edge proxy should also limit invalid-credential traffic and raw
request body size because those requests cannot safely be assigned to a trusted
user bucket.

## Phase 8: Structured Logging and Request Correlation

Phase 8 makes one request traceable through the API without putting private RAG
content into logs. Every HTTP response, including health checks, authentication
failures, rate-limit responses, and validation errors, receives an
`X-Request-ID` header. A client may send its own UUID in the same header; the API
canonicalizes it. Missing or malformed values are replaced with a new UUIDv4,
so an untrusted header cannot inject forged text into a log line.

Application events are emitted as one JSON object per line. A completed request
contains the timestamp, severity, event name, request ID, HTTP method, path,
status code, and duration. After successful authentication it also contains the
trusted user ID and resolved session ID. This makes it possible to search all
events for one request or session in a log platform.

### Scope

- [x] A UUID request ID on every API response
- [x] Safe acceptance and canonicalization of client UUIDs
- [x] Replacement of malformed correlation headers
- [x] Machine-readable JSON application logs
- [x] Request duration and response status metadata
- [x] Correlation with trusted user and resolved session IDs
- [x] Allowlisted log fields instead of arbitrary object serialization
- [x] Generic exception types without exception messages or tracebacks
- [x] Runtime-selectable bounded log severity
- [x] Tests proving questions, answers, API keys, headers, query strings, and
  retrieved content are not logged

Configure the minimum application severity in `.env`:

```dotenv
RAG_LOG_LEVEL=INFO
```

The accepted values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
Compose injects this setting only into the API container. The setting is not
baked into the Docker image.

For example, the response header and matching log event may look like this:

```text
X-Request-ID: 11446d6e-4948-4c80-9816-c752d1416fc5
```

```json
{"timestamp":"2026-08-29T12:00:00.000Z","level":"INFO","logger":"rag.api","event":"request.completed","request_id":"11446d6e-4948-4c80-9816-c752d1416fc5","user_id":"local-user","session_id":"b07e89c8-bdb9-456f-a255-718b48afbd43","method":"POST","path":"/v1/query","status_code":200,"duration_ms":84.213}
```

The logger intentionally excludes authorization headers, API keys, request and
response bodies, URL query strings, questions, answers, citations, retrieved
chunks, raw exception messages, and tracebacks. This reduces exposure, but user
and session IDs are still operational metadata and should be retained only as
long as necessary.

Request correlation is not distributed tracing. It connects events inside this
API process, but it does not yet create spans across Groq, vector storage, or
future services. W3C trace propagation and OpenTelemetry remain a later phase.

## Docker Compose Workflow

The application supports both one-shot container jobs and a long-running API
service. Build the runtime image once:

```bash
docker compose build
```

Build or update the index from the host's read-only `data/raw/` directory:

```bash
docker compose run --rm index
```

Indexing is incremental. A collection-specific manifest is stored beside
Chroma in the `rag-chroma` volume. Unchanged source files reuse their existing
chunks, changed or missing sources are re-embedded, deleted sources are removed,
and changes to the embedding model or chunking configuration trigger a safe
full rebuild.

Only one process may update a collection at a time. The index command holds a
collection-specific operating-system lock for the complete Chroma-and-manifest
transaction. A competing command fails immediately with a clear error instead
of racing the active writer; optionally make it wait for up to 30 seconds:

```bash
docker compose run --rm index --lock-timeout-seconds 30
```

The `.lock` file is only a rendezvous point. A crash automatically releases its
kernel-held lock, so an old file does not block future indexing.

Ask a question using the same persistent index:

```bash
docker compose run --rm ask "What is reciprocal rank fusion?"
```

Run the Phase 3 quality gate and print its persisted report:

```bash
docker compose run --rm evaluate
docker compose run --rm report
```

Compose reads `GROQ_API_KEY`, `GROQ_MODEL`, `RAG_API_KEY_HASHES`, and
`RAG_DOCUMENT_PERMISSIONS` from the local `.env` file, along with the optional
rate-limit and log-level settings. It injects authentication hashes, document
policy, API limits, and application logging configuration only into the API and
never adds them to the image. Named volumes persist
the Chroma index, downloaded model cache, and evaluation report across
disposable containers.

The Dockerfile selects `requirements-linux-amd64.lock` or
`requirements-linux-arm64.lock` from BuildKit's target architecture. Compose
does not force a platform, so it builds and runs natively on both Intel/AMD
Linux and ARM64 machines such as Apple Silicon.

## Container Image CI

`.github/workflows/container-image.yml` verifies every pull request on native
AMD64 and ARM64 runners by building the dependency-locked Docker test target,
building the runtime image, and scanning it for high and critical
vulnerabilities. A JSON report for each architecture is retained as a workflow
artifact. Fixable high or critical findings fail the check; all findings remain
visible in the reports.

After the same verification succeeds on a trusted push to `main` or a `v*`
release tag, GitHub Actions publishes a native `linux/amd64` + `linux/arm64`
image to `ghcr.io/raghava4272/rag-system`. Every publication receives an
immutable full-commit tag (`sha-<commit>`). Main also updates `main` and
`latest`; version tags such as `v1.2.3` produce `1.2.3`, `1.2`, and `latest`.

Published images include BuildKit provenance and an SBOM, and the published
manifest digest receives a GitHub build-provenance attestation. Publishing uses
the repository-scoped `GITHUB_TOKEN`; no registry password or Groq key is
required. GitHub Actions and the Docker base image are checked weekly by
Dependabot while workflow actions remain pinned to immutable commit SHAs.

Pull a public package after the first successful `main` build:

```bash
docker pull ghcr.io/raghava4272/rag-system:latest
```

If the package is private, authenticate to GHCR or change its visibility in the
repository's Packages settings before pulling it anonymously.

To inspect the persistent resources:

```bash
docker volume ls --filter label=com.docker.compose.project=rag-system
```

Deleting the `rag-system_rag-chroma` volume deletes the built index, so do not
remove Compose volumes unless you intend to rebuild it.
