# ICP Taxonomy Policy Evaluation Framework (LLM-as-a-Judge)
**Author:** Rishabh Bhawsar  
**Email:** rishabhbhawsar53@gmail.com | **LinkedIn:** https://www.linkedin.com/in/rishabh-bhawsar-409098262/


A production-style backend service that uses a Large Language Model as an automated **policy compliance judge** for Ideal Customer Profile (ICP) product taxonomy classification — the kind of system a FinTech risk, compliance, or growth-ops team would use to validate whether merchant/product listings conform to region-specific policy rules, at scale, without a human reviewing every single row.

This repo is a from-scratch, incrementally-committed engineering project. Every commit in the history represents a real design decision, not a single generated dump — the commit log itself is meant to be read.

---

## 1. The Problem This Solves

FinTech and marketplace platforms operating across multiple countries must classify incoming products/merchants against an **Ideal Customer Profile taxonomy** — a rulebook that differs by locale (e.g., what counts as a "restricted financial product" in Germany differs from Brazil or Singapore). Doing this manually does not scale past a few hundred SKUs a day, and naive LLM prompting without structural guarantees produces inconsistent, unparseable, or silently-wrong output that downstream systems can't trust.

This project builds a **judge service**: given a product/merchant description and a locale, it asks an LLM to classify the entry against that locale's policy taxonomy, and — critically — *guarantees* the response is structurally valid, auditable, cached, and fast enough to run in a real pipeline.

## 2. Why This Is an Interesting Engineering Problem (not just "call the OpenAI API")

| Naive approach | Why it breaks in production | What this project does instead |
|---|---|---|
| `response = llm.chat(prompt)` then `json.loads(response)` | LLMs occasionally emit prose, markdown fences, or malformed JSON — this throws in prod | Pydantic-enforced structured output (JSON mode + schema validation) with typed failure handling |
| One request at a time in a loop | A 5,000-row batch takes minutes; the event loop blocks on network I/O the whole time | `asyncio`-based concurrent request pooling via `asyncio.gather`, with bounded concurrency (semaphores) so we don't get rate-limited |
| Re-classify everything on every run | Wastes tokens and money on inputs you've already judged | A caching layer (Redis-shaped interface, simulated locally) keyed on a hash of (input, locale, policy version) |
| No record of what the model decided or why | No audit trail — a real compliance system needs one | Every judgment is persisted to a SQLite ledger with timestamp, input hash, raw + parsed output, and policy version, so any decision can be traced back |
| Trust whatever the model says | LLMs are not deterministic; some fraction of outputs will be malformed or low-confidence | Explicit schema-adherence and confidence-threshold tracking, so structural failure and business-logic uncertainty are handled as distinct, visible failure modes rather than silently swallowed |

## 3. Architecture

```
                        ┌─────────────────────────┐
                        │   FastAPI (async)        │
                        │   /evaluate, /batch,     │
                        │   /taxonomy, /health      │
                        └────────────┬─────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │      Evaluator Service         │
                     │  - builds locale-aware prompt   │
                     │  - checks cache before calling  │
                     │  - dispatches to OpenAI client   │
                     │  - validates response (Pydantic)│
                     └───────┬───────────────┬─────────┘
                             │               │
                 ┌───────────▼───┐   ┌───────▼────────────┐
                 │ OpenAI Client  │   │  Cache Layer        │
                 │ (async, JSON   │   │  (Redis-shaped;      │
                 │  mode, retry/  │   │  local dict/SQLite    │
                 │  backoff)      │   │  fallback for dev)    │
                 └────────────────┘   └─────────────────────┘
                             │
                     ┌───────▼─────────────┐
                     │   SQLite Ledger       │
                     │  every judgment,       │
                     │  input hash, output,   │
                     │  policy version, time  │
                     └───────────────────────┘
```

**Request flow for a single evaluation:**
1. Client sends a product/merchant description + target locale to `/evaluate`.
2. The evaluator computes a cache key (hash of input + locale + policy version). If cached, returns instantly.
3. On a cache miss, it builds a structured prompt referencing that locale's taxonomy policy and calls the OpenAI client asynchronously.
4. The raw model response is parsed and validated against a strict Pydantic schema. A structurally invalid response triggers a bounded retry with a corrective prompt before it's surfaced as a typed error.
5. The validated judgment is written to the SQLite ledger and the cache, then returned to the client.
6. `/batch` fans this same flow out across many inputs concurrently via `asyncio.gather`, bounded by a semaphore so we respect API rate limits.

## 4. Tech Stack & Why

- **FastAPI** — async-native, so I/O-bound LLM calls don't block the server; automatic OpenAPI/Swagger docs give recruiters a live, interactive API to click through without reading code.
- **Pydantic** — the data contract between "what the LLM said" and "what the rest of the system is allowed to assume is true." Turns hallucinated formatting into a caught, typed exception instead of a runtime crash three services downstream.
- **OpenAI SDK (JSON mode)** — constrains the model's output space at generation time rather than hoping a prompt instruction is obeyed.
- **asyncio** — the concurrency primitive that makes batch evaluation fast without needing a separate task queue for a project at this scale.
- **SQLite** — zero-infrastructure persistence that still gives you a real, queryable audit ledger; appropriate for a portfolio-scale deployment (and a deliberate, explainable choice over provisioning Postgres for a single-instance demo).
- **Redis (simulated locally)** — the interface is written against Redis semantics (`get`/`set`/TTL) so the caching layer is a drop-in swap to real Redis in a production deployment; locally it runs against an in-memory/SQLite-backed stand-in so the whole thing works on a free-tier host with no external services.

## 5. Scope

- 53 product taxonomy policies mapped across 17 locale configurations (see `src/models/taxonomy.py` once built).
- Single and batch evaluation endpoints.
- A benchmark/eval harness (added later in this build) that measures — rather than assumes — schema adherence rate, sync-vs-async latency delta, and classification precision/false-positive rate against a labeled validation set. **Numbers in any future version of this README are pulled directly from that harness's output, not asserted in advance.**

## 6. Deployment

Designed to run on a free-tier host (Render free Web Service or Hugging Face Spaces via Docker) with SQLite as the only persistence dependency, exposing interactive Swagger docs at `/docs` so anyone evaluating this project can exercise the API directly in-browser.

## 7. Project Status

🚧 Actively being built incrementally, file by file, with each stage committed separately. See commit history for build order: config/schema foundations → OpenAI client → evaluator service → API layer → concurrency + caching → benchmark harness.