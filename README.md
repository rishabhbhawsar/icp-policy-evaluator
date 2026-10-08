# ICP Taxonomy Policy Evaluation Framework (LLM-as-a-Judge)

A production-oriented backend that uses a Large Language Model as an automated **policy compliance judge**, evaluating merchant and product onboarding descriptions against a locale-scoped regulatory taxonomy.

The system is built around the structural guarantees — schema validation, caching, retry/backoff, and an audit ledger — that a straightforward "call the LLM and hope" pipeline lacks.

---

## 1. Executive Abstract & Problem Statement

Running compliance decisions through a raw LLM call, one request at a time, introduces three costs that compound in production:

- **Token cost.** Re-evaluating the same onboarding description on every retry, re-submission, or duplicate request burns money on inputs that have already been judged.
- **Latency.** A judge call that synchronously blocks on network I/O does not scale past a handful of concurrent requests before the event loop backs up.
- **Trust.** LLM output is not guaranteed to be structurally valid JSON, semantically consistent with itself, or free of fabricated metadata — a model asked to report its own name or an audit ID will readily invent one.

This framework addresses all three concerns.

A content-hashed local cache eliminates redundant judge calls. `asyncio`-bounded concurrency keeps batch throughput predictable under a configurable request-in-flight ceiling. Every judge response is validated against a strict Pydantic contract before it is trusted anywhere downstream.

---

## 2. Core Technical Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | **FastAPI** | Async-native; blocking judge calls do not stall the event loop. Auto-generates Swagger/OpenAPI documentation. |
| Data contracts | **Pydantic V2** | Strict `extra="forbid"` schemas turn LLM formatting drift into a caught, typed exception instead of a silent bad value downstream. Cross-field validators enforce classification/risk-level consistency. |
| LLM orchestration | **OpenAI SDK (`AsyncOpenAI`)** | Uses an OpenAI-compatible endpoint, making the provider layer interchangeable. |
| Retry/backoff | **Tenacity** | Exponential backoff with jitter on transient upstream failures such as 429, 5xx, and timeouts. |
| Persistence | **SQLite (`aiosqlite`), WAL mode** | Zero-infrastructure audit ledger and cache suitable for free-tier deployment. |
| Configuration | **pydantic-settings** | Typed `.env`-driven configuration. Secrets and provider configuration stay outside application code. |
| Current LLM provider | **Groq** | OpenAI-compatible gateway. Provider, model, and endpoint are configuration-driven. |
| Deployment | **Render / Hugging Face Spaces** | SQLite avoids requiring an external managed database. |
| ASGI server | **Uvicorn** | Standard ASGI server for local development and deployment. |

---

## 3. Asynchronous Request Flow

```text
Client
  │
  │ POST /evaluate
  │ FastAPI + Pydantic validation
  ▼
PolicyEvaluator.evaluate()
  │
  ├─ 1. Local input guardrail
  │      Reject empty/whitespace descriptions
  │      BEFORE any network egress
  │      → 422, zero judge calls, zero cost
  │
  ├─ 2. TaxonomyRepository.get(policy_id, locale)
  │      └─ Not found → 404, no judge call
  │
  ├─ 3. cache_key =
  │      sha256(
  │        policy_id +
  │        locale +
  │        policy_version +
  │        description
  │      )
  │
  │      CacheBackend.get(cache_key)
  │      [SQLite, WAL mode]
  │
  │      ├─ HIT
  │      │    → Return cached result
  │      │    → No judge call
  │      │    → No new ledger row
  │      │
  │      └─ MISS
  │           → Continue
  │
  ├─ 4. OpenAIJudgeClient.evaluate()
  │
  │      ├─ Builds a locale-scoped system prompt
  │      │   from the policy's own rules
  │      │
  │      ├─ Tenacity
  │      │   Exponential backoff + jitter on transient
  │      │   provider errors
  │      │
  │      ├─ Narrow judgment-only JSON schema
  │      │   Model reports only:
  │      │   classification
  │      │   confidence
  │      │   risk_level
  │      │   violated_rules
  │      │   reasoning
  │      │
  │      ├─ policy_id, locale, and model_name are
  │      │   injected from application code
  │      │   and are never trusted from model output
  │      │
  │      ├─ Local retry if provider JSON fails
  │      │   schema validation
  │      │
  │      └─ OpenAI-compatible endpoint
  │           → Groq
  │           → underlying model
  │
  ├─ 5. LedgerWriter.record()
  │      Append-only audit row
  │
  ├─ 6. CacheBackend.set()
  │      TTL-bounded cache write
  │
  ▼
EvaluationResult
  │
  ▼
FastAPI response_model
  │
  ▼
Client
```

### Batch Evaluation

`POST /batch` follows the same evaluation pipeline for every item, but fans requests out using `asyncio.gather()`.

Concurrency is bounded by an `asyncio.Semaphore(batch_concurrency_limit)`.

This prevents a large batch from creating an uncontrolled number of simultaneous LLM requests.

A single item's failure is returned as a typed per-item failure rather than aborting the entire batch.

Therefore, one bad row in a batch of 100 does not lose the other 99 results.

Two consecutive runs of the integration harness against unchanged input produced zero new judge calls and zero new ledger rows on the second run.

After clearing the cache, a fresh pair of judge calls and ledger writes confirmed both branches of the cache logic against a live provider.

---

## 4. Local Defensive Guardrails

### Input Validation

Input is validated before any provider request.

A whitespace-only description never reaches the network.

This means:

- Zero judge-client calls
- Zero token cost
- Immediate `422` response

### Strict Pydantic Contracts

Every schema uses strict validation, including:

- `extra="forbid"`
- Regex-constrained IDs
- `confidence: float` bounded between `0.0` and `1.0`
- Cross-field validation
- Classification/risk consistency checks

Examples of invalid states rejected by the schema:

- `COMPLIANT` result containing violated rule IDs
- `PROHIBITED` risk rule defining a disclosure escape hatch
- `COMPLIANT` classification carrying `HIGH` risk

### Failure-Aware Retry Strategy

Retries are classified by failure type rather than blindly retrying everything.

#### Transient provider failures

These include:

- Rate limits
- Timeouts
- HTTP 5xx errors
- Connection failures

These receive exponential backoff with jitter, capped at 5 attempts.

After exhaustion, the error is re-raised instead of being retried indefinitely.

#### Structural LLM failures

Examples include:

- Safety refusal
- Token-truncated response
- Empty response body
- Invalid JSON/schema structure

These receive a local retry with a strengthened prompt because the underlying request may still be recoverable.

### HTTP Error Mapping

Typed application exceptions are mapped to clean HTTP responses:

| Error | HTTP Status |
|---|---:|
| Taxonomy/policy not found | `404` |
| Invalid request/input | `422` |
| Provider/upstream failure | `502` |

The client receives a machine-readable error body instead of a raw traceback.

### Known Limitation

`/evaluate` and `/batch` currently have **no authentication or rate limiting**.

That is acceptable for a demo behind a free-tier Swagger UI, but it is a real security and scalability gap that would need to be closed before handling production traffic.

---

## 5. How to Run & Reproduce

### 1. Clone the repository

```bash
git clone <your-repo-url> icp-policy-evaluator
cd icp-policy-evaluator
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your_groq_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b
DATABASE_PATH=compliance.db
```

### 5. Run the provider preflight check

This confirms that the configured key, endpoint, and model are working:

```bash
python -m scripts.check_provider
```

### 6. Run the integration smoke test

This performs real judge calls and writes to the real ledger:

```bash
python -m tests.test_harness
```

### 7. Run the accuracy benchmark

```bash
python -m tests.evaluation_runner
```

### 8. Start the API server

```bash
uvicorn src.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## 6. Rate Limit Considerations

Groq's free tier enforces an **8,000 tokens/minute limit per model**.

The benchmark's default `batch_concurrency` may trigger `429` responses during a full run.

The retry layer handles these using exponential backoff.

For faster local iteration, reduce concurrency:

```env
BATCH_CONCURRENCY_LIMIT=2
```

---

## 7. Measured Baseline

Results from `tests/evaluation_runner.py` against a live provider:

| Metric | Value |
|---|---:|
| Precision | **1.000** |
| Recall | **0.818** |
| F1-Score | **0.900** |
| Accuracy | **0.929** |

### Confusion Matrix

```text
TP = 9
TN = 17
FP = 0
FN = 2

Total = 28
```

---

## 8. Benchmark Composition

The benchmark contains **28 cases**:

- **6** shell-structure cases
  - Multi-tier ownership
  - Trusts
  - Nominees

- **5** ambiguous neobank cases
  - Ownership never mentioned

- **4** explicit-decliner cases
  - States outright that controllers are not identified

- **5** surface-risky-but-compliant cases
  - Business sounds risky
  - Disclosure is nevertheless adequate

- **3** trap cases
  - Surface cue contradicts the correct answer

- **3** adversarial cases
  - Prompt injection
  - Field injection
  - Self-contradictory input

- **2** public-company exemption cases
  - Owner is a listed company

---

## 9. Benchmark Methodology & Limitations

### Scope

The benchmark uses:

- Single policy: `KYC-014`
- Policy type: beneficial-ownership disclosure
- Single locale: `US`
- Single model/provider snapshot
- Dataset size: `n=28`

### Provider Snapshot

The reported metrics were captured against:

```text
Provider: Groq
Model: openai/gpt-oss-120b
```

Re-running against a different OpenAI-compatible provider is a configuration change.

The expected output structure should remain similar, but exact metrics may change.

### Dataset Size

`n=28` is large enough to show a meaningful spread of outcomes, but it is not large enough to provide statistically tight confidence intervals.

One flipped case changes accuracy by approximately **3.5 percentage points**.

### False Positive Interpretation

Zero observed false positives does **not** mean the system has a proven zero false-positive rate.

With zero false positives in 17 negative-labeled trials, a true false-positive rate as high as roughly 17% could still be statistically consistent with the observed sample.

The defensible claim is:

> In this benchmark, the judge never approved a case that should have been rejected.

It should not be interpreted as a general guarantee.

### False Negative Analysis

Both false negatives occurred in the **surface-risky-but-compliant** category.

The judge escalated these cases to:

- `NON_COMPLIANT`
- or `REQUIRES_HUMAN_REVIEW`

even though the underlying disclosure was adequate.

This reflects a conservative bias where category risk is weighted alongside disclosure adequacy.

For a compliance system, this is generally the preferable error direction, but it remains a known limitation.

### Adversarial Testing

The benchmark contains three adversarial cases:

1. Prompt injection
2. Field injection
3. Self-contradictory input

The judge correctly rejected all three.

Injected instructions or pre-filled fields did not override the judge's own determination.

---

## 10. Architecture Principles

The project is designed around several production-oriented principles.

### 1. Never Trust LLM-Generated Metadata

The model should not be responsible for generating authoritative metadata such as:

- Policy ID
- Locale
- Model name
- Audit ID

These values are injected by application code.

### 2. Separate Network Retries from Structural Retries

Provider failures and invalid model output are different failure classes.

```text
Provider failure
      ↓
Tenacity retry
      ↓
Exponential backoff + jitter
```

versus:

```text
Invalid model response
      ↓
Local schema validation
      ↓
Strengthened prompt
      ↓
Single structural retry
```

### 3. Cache Deterministically

The cache key is derived from:

```text
policy_id
+
locale
+
policy_version
+
description
```

and hashed with SHA-256.

This means identical evaluations can be returned without another LLM request.

### 4. Keep Batch Concurrency Bounded

Instead of allowing an arbitrary number of simultaneous requests:

```text
Batch
 ├── Request 1
 ├── Request 2
 ├── Request 3
 ├── ...
 └── Request N
```

the semaphore enforces a controlled number of in-flight requests:

```text
Batch
      │
      ▼
Concurrency Semaphore
      │
      ├── Worker 1
      ├── Worker 2
      └── Worker N
```

This makes throughput and provider pressure predictable.

---

## 11. Roadmap

### Security

- Authentication and rate limiting on `/evaluate` and `/batch`
- API keys with per-key token buckets
- Request signing for higher-trust clients

### Infrastructure

- Migrate cache from SQLite to Redis
- Support cross-instance caching
- Increase cache write throughput
- Deploy to AWS EC2 with Docker

### Frontend

- Virtualized list rendering for batch results
- Prevent large evaluation runs from bloating the DOM
- Improve scrolling performance for hundreds of rows

### CI/CD

- GitHub Actions integration
- Run benchmark suite on every push
- Report benchmark metrics automatically as PR comments

---

## 12. Project Highlights

This project demonstrates practical implementation of:

- **FastAPI**
- **Async Python**
- **Pydantic V2**
- **OpenAI-compatible LLM APIs**
- **Groq**
- **LLM-as-a-Judge**
- **Structured LLM output**
- **Schema validation**
- **Prompt engineering**
- **Tenacity retry/backoff**
- **Exponential backoff with jitter**
- **SQLite**
- **aiosqlite**
- **WAL mode**
- **Content-addressed caching**
- **TTL caching**
- **Async concurrency control**
- **`asyncio.Semaphore`**
- **Batch processing**
- **Audit ledgers**
- **LLM evaluation**
- **Adversarial testing**
- **Prompt-injection resistance**
- **FastAPI error handling**
- **Production-oriented API design**

---

## 13. Author

**Rishabh Bhawsar**

- GitHub: [github.com/rishabhbhawsar](https://github.com/rishabhbhawsar)
- LinkedIn: [linkedin.com/in/rishabh-bhawsar-409098262](https://linkedin.com/in/rishabh-bhawsar-409098262)

---

## 14. License

MIT License.

You are free to use, modify, and distribute this software, including for commercial purposes.

Keep the copyright notice and permission notice in substantial copies of the software.

No warranty is provided.