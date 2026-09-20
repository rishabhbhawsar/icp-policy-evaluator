# ICP Taxonomy Policy Evaluation Framework (LLM-as-a-Judge)

A production-oriented backend that uses a Large Language Model as an automated **policy compliance judge**, evaluating merchant and product onboarding descriptions against a locale-scoped regulatory taxonomy. The system is built around the structural guarantees — schema validation, caching, retry/backoff, and an audit ledger — that a straightforward "call the LLM and hope" pipeline lacks.

---

## 1. Executive Abstract & Problem Statement

Running compliance decisions through a raw LLM call, one request at a time, introduces three costs that compound in production:

- **Token cost.** Re-evaluating the same onboarding description on every retry, re-submission, or duplicate request burns money on inputs that have already been judged.
- **Latency.** A judge call that synchronously blocks on network I/O does not scale past a handful of concurrent requests before the event loop backs up.
- **Trust.** LLM output is not guaranteed to be structurally valid JSON, semantically consistent with itself, or free of fabricated metadata — a model asked to report its own name or an audit ID will readily invent one.

This framework addresses all three concerns. A content-hashed local cache eliminates redundant judge calls, and this is demonstrated rather than assumed (see §3). `asyncio`-bounded concurrency keeps batch throughput predictable under a configurable request-in-flight ceiling. Every judge response is validated against a strict Pydantic contract before it is trusted anywhere downstream.

## 2. Core Technical Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | **FastAPI** | Async-native; blocking judge calls do not stall the event loop. Auto-generates the Swagger/OpenAPI docs this repo is meant to be explored through. |
| Data contracts | **Pydantic V2** | Strict, `extra="forbid"` schemas turn LLM formatting drift into a caught, typed exception instead of a silent bad value three layers downstream. |
| LLM orchestration | **OpenAI SDK (`AsyncOpenAI`)**, pointed at an OpenAI-compatible endpoint | Provider-agnostic by construction. The SDK's exception hierarchy (`RateLimitError`, `APIConnectionError`, etc.) is raised from HTTP and transport behavior, not from which server answers, so the same retry logic works unmodified against OpenAI, **OpenRouter** (the provider currently configured), or any other OpenAI-compatible gateway. |
| Retry/backoff | **Tenacity** | Exponential backoff with jitter on transient upstream failures (429/5xx/timeout), capped and re-raised on exhaustion rather than retried forever. |
| Persistence | **SQLite (via `aiosqlite`), WAL mode** | Zero-infrastructure, free-tier-deployable audit ledger and cache, sharing one file across two connections without one blocking the other. |
| Config | **pydantic-settings** | Typed, `.env`-driven settings (API key as `SecretStr`, base URL, model, cache TTL, batch concurrency). No credentials or endpoints hardcoded in application code. |
| Current LLM provider | **OpenRouter** (OpenAI-compatible gateway) | Free-tier access to a broad model catalog behind one API shape; swappable via a single config value rather than a code change. |
| Deployment target | **Render** (free Web Service) or **Hugging Face Spaces** (Docker) | SQLite requires no external managed database, keeping the whole stack inside a free tier. |
| ASGI server | **Uvicorn** | Standard ASGI server for running the FastAPI app in both local development and deployment. |

## 3. Asynchronous Request Flow

```
Client
  │  POST /evaluate  (FastAPI route, Pydantic-validated request body)
  ▼
PolicyEvaluator.evaluate()
  │
  ├─ 1. Local input guardrail: reject empty/whitespace descriptions
  │       BEFORE any network egress  →  422, zero judge calls, zero cost
  │
  ├─ 2. TaxonomyRepository.get(policy_id, locale)
  │       └─ not found → 404, no judge call
  │
  ├─ 3. cache_key = sha256(policy_id + locale + policy_version + description)
  │       └─ CacheBackend.get(cache_key)   [SQLite, WAL mode]
  │             ├─ HIT  → return cached result. No judge call. No new ledger row.
  │             └─ MISS → continue
  │
  ├─ 4. OpenAIJudgeClient.evaluate()
  │       ├─ builds a locale-scoped system prompt from the policy's own rules
  │       ├─ Tenacity: exponential backoff + jitter on transient provider errors
  │       │     (RateLimitError / APIConnectionError / APITimeoutError / InternalServerError)
  │       ├─ narrow judgment-only JSON schema — the model reports only
  │       │     classification/confidence/risk_level/violated_rules/reasoning;
  │       │     policy_id, locale, and model_name are injected from code,
  │       │     never trusted to the model's self-report
  │       ├─ bounded local retry if the provider's JSON doesn't match the
  │       │     schema on the first attempt (distinct from the network retry above)
  │       └─ → OpenAI-compatible endpoint (OpenRouter) → underlying model
  │
  ├─ 5. LedgerWriter.record()   — append-only audit row (aiosqlite)
  ├─ 6. CacheBackend.set()      — TTL-bounded cache write (aiosqlite, same db file)
  ▼
EvaluationResult → FastAPI response_model → client
```

**Batch path (`POST /batch`):** the same six steps, fanned out per item via `asyncio.gather()`, bounded by an `asyncio.Semaphore(batch_concurrency_limit)`. This was verified under test to cap concurrent in-flight judge calls at exactly the configured limit, rather than merely intended to. A single item's failure (bad input, provider error) is returned as a typed per-item failure, not raised — one bad row in a batch of 100 does not lose the other 99 results.

**Cache effectiveness is demonstrated, not assumed.** Two consecutive runs of the integration harness against unchanged input produced zero new judge calls and zero new ledger rows on the second run. A full fresh pair of judge calls and ledger writes followed immediately after the cache was cleared. Both branches of the cache logic were observed live against a real provider, not mocked.

## 4. Local Defensive Guardrails

- **Input validated before egress.** A whitespace-only description never reaches the network — confirmed under test (zero judge-client calls made) before it was ever trusted in production.
- **Strict Pydantic contracts everywhere.** `extra="forbid"` on every schema catches hallucinated fields; regex-constrained IDs; a bounded `confidence: float` (0.0–1.0); cross-field validators (a `COMPLIANT` result cannot simultaneously carry violated rule IDs; a `PROHIBITED`-risk rule cannot define a disclosure escape hatch).
- **Retry classified by failure type, not blanket-retried.** Transient upstream errors (rate limits, timeouts, 5xx) get exponential backoff with jitter, capped at 5 attempts, then re-raised as a real error rather than retried indefinitely. Structural failures (a safety refusal, a token-truncated response) are *not* retried, since retrying does not change a guaranteed-repeat outcome.
- **Typed exception → HTTP status mapping**, so a client receives a clean `404`/`422`/`502` with a machine-readable error body instead of a raw traceback.

**Known limitation, stated plainly rather than omitted:** `/evaluate` and `/batch` currently have no authentication or rate limiting. That is an acceptable posture for a portfolio demo behind a free-tier Swagger UI, but it is a real gap that would need closing before any actual production traffic.

## 5. How to Run & Reproduce

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> icp-policy-evaluator
cd icp-policy-evaluator

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment -- create a .env file in the repo root:
#    OPENAI_API_KEY=sk-or-v1-your_actual_key
#    OPENAI_BASE_URL=https://openrouter.ai/api/v1
#    OPENAI_MODEL=google/gemini-2.5-flash
#    DATABASE_PATH=compliance.db

# 5. Preflight check -- confirms the key/base_url/model actually work
#    before running anything else
python -m scripts.check_provider

# 6. Run the integration smoke test (real judge calls, real ledger)
python -m tests.test_harness

# 7. Run the accuracy benchmark
python -m tests.evaluation_runner

# 8. Run the API server
uvicorn src.main:app --reload
# then open http://127.0.0.1:8000/docs
```

## 6. Measured Baseline (not a placeholder)

The numbers below are from an actual run of `tests/evaluation_runner.py` against a live provider, not target figures decided in advance. See the confusion matrix for the full methodology.

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.778 |
| F1-Score | 0.875 |
| Accuracy | 0.913 |

**Confusion matrix:** TP=7, TN=14, FP=0, FN=2 (n=23)

**Methodology and honest limits:**
- Single policy (`KYC-014`, beneficial-ownership disclosure), single locale (`US`), single model/provider snapshot.
- n=23 is enough to see a real spread of outcomes — it is not a 4-case coin flip — but it is **not** large enough for a statistically tight confidence interval. One flipped case moves accuracy by roughly 4 points.
- **Zero observed false positives is not the same claim as a proven zero false-positive rate.** With 0 events in 14 negative-labeled trials, a true FP rate as high as roughly 20% would still be statistically consistent with what was observed. The honest claim is: *in this test set, the judge never approved a case it should have rejected* — not that this property is proven to hold in general.
- Both false negatives were individually inspected, not just counted. One (`shell_trust_fully_disclosed`) arguably reflects the judge applying a stricter beneficial-ownership standard than the test label assumed. The other (`risky_surface_frontier_credit_disclosed`) was the one case in its group relying on self-reported disclosure with no independent verification clause, suggesting the judge weights corroborated evidence over bare assertion.
- This baseline should be re-run and expanded (more cases per class, ideally scored by more than one rater) before being cited as a fixed, load-bearing number.

## 7. Roadmap: Planned Frontend Layer (Not Yet Implemented)

Everything below is a stated future direction, not existing functionality. No frontend code exists in this repo yet. Concepts are drawn from Akshay Saini's Frontend System Design (FSD) framing.

- **Virtualized list rendering** for the batch-results view, so a large evaluation run (hundreds of rows) does not bloat the DOM or degrade scroll performance.
- **Debounced input and request deduplication** on any interactive evaluation form, so rapid keystrokes do not fan out redundant calls to `/evaluate`.
- **Secure session management** — HttpOnly cookie storage and JWT rotation — for whatever auth layer eventually sits in front of these currently-unauthenticated endpoints (see §4's known limitation).

---

*Built incrementally, commit by commit, as a portfolio project demonstrating production backend design for AI-orchestrated systems.*