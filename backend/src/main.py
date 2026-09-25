"""src/main.py

FastAPI composition root. Reads all configuration through get_settings()
(src/core/config.py) rather than raw os.environ lookups. Wires
OpenAIJudgeClient + PolicyEvaluator to the real aiosqlite-backed
LedgerWriter/CacheBackend (persistence.py). No real policy-registry store
exists yet, so TaxonomyRepository remains an in-memory stand-in seeded with
demo data -- swap it for a real repository without touching evaluator.py or
these routes.

Error mapping:
  EmptyInputError         -> 422  (syntactically valid request, semantically empty content)
  PolicyNotFoundError     -> 404  (referenced resource does not exist)
  JudgeRefusalError       -> 422  (model declined to classify the content)
  JudgeSchemaViolationError -> 422  (model output was valid JSON but failed schema validation)
  JudgeTruncationError    -> 500  (server-side token-budget misconfiguration, not client fault)
  OpenAIError             -> 502  (upstream dependency failure, retries already exhausted)
  Exception (catch-all)   -> 500  (never leak raw tracebacks to the client)
"""

from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import get_settings
from src.models.taxonomy import (
    ComplianceRule,
    EvaluationResult,
    ICPTaxonomy,
    PolicyCategory,
    RiskLevel,
    SupportedLocale,
)
from src.services.evaluator import (
    EmptyInputError,
    EvaluationFailure,
    PolicyEvaluator,
    PolicyNotFoundError,
)
from src.services.openai_client import (
    JudgeRefusalError,
    JudgeSchemaViolationError,
    JudgeTruncationError,
    OpenAIJudgeClient,
)
from src.services.persistence import SQLiteCacheBackend, SQLiteLedgerWriter

logger = logging.getLogger(__name__)

_POLICY_ID_PATTERN = r"^[A-Z]{2,10}-\d{3}$"

# --- taxonomy repository: still in-memory. No SQLite-backed policy registry has been
# built yet (Step 6 covered ledger + cache only); this is a known open item, not an
# oversight -- see module docstring. ---


class InMemoryTaxonomyRepository:
    def __init__(self, policies: list[ICPTaxonomy]) -> None:
        self._by_key = {(p.policy_id, p.locale): p for p in policies}

    async def get(self, policy_id: str, locale: SupportedLocale) -> ICPTaxonomy | None:
        return self._by_key.get((policy_id, locale))


_SEED_POLICIES = [
    ICPTaxonomy(
        policy_id="KYC-014",
        category=PolicyCategory.KYC_AML,
        locale=SupportedLocale.US,
        version=1,
        effective_date=date(2026, 1, 1),
        rules=[
            ComplianceRule(
                rule_id="KYC-014",
                description="Beneficial ownership must be disclosed for entities holding >25% equity.",
                risk_level=RiskLevel.HIGH,
                required_disclosures=["beneficial_owner_identity"],
            )
        ],
    ),
]


# --- API request/response contracts ---


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_description: str = Field(..., min_length=1, max_length=5000)
    policy_id: str = Field(..., pattern=_POLICY_ID_PATTERN)
    locale: SupportedLocale


class BatchEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EvaluateRequest] = Field(..., min_length=1, max_length=100)


class BatchItemResult(BaseModel):
    index: int
    status: Literal["success", "failure"]
    result: EvaluationResult | None = None
    error: str | None = None


class BatchEvaluateResponse(BaseModel):
    results: list[BatchItemResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    timestamp: datetime
    openai_configured: bool


# --- composition root ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    judge_client = OpenAIJudgeClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        max_tokens=settings.judge_max_tokens,
    )
    ledger = await SQLiteLedgerWriter.create(settings.database_path)
    cache = await SQLiteCacheBackend.create(settings.database_path)

    app.state.settings = settings
    app.state.judge_client = judge_client
    app.state.ledger = ledger
    app.state.cache = cache
    app.state.evaluator = PolicyEvaluator(
        judge_client=judge_client,
        taxonomy_repo=InMemoryTaxonomyRepository(_SEED_POLICIES),
        cache=cache,
        ledger=ledger,
        cache_ttl_seconds=settings.cache_ttl_seconds,
        batch_concurrency=settings.batch_concurrency_limit,
    )

    yield

    await judge_client.aclose()
    await cache.close()
    await ledger.close()


app = FastAPI(title="ICP Taxonomy Policy Evaluator", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_evaluator(request: Request) -> PolicyEvaluator:
    return request.app.state.evaluator


# --- exception handlers ---


@app.exception_handler(EmptyInputError)
async def handle_empty_input(request: Request, exc: EmptyInputError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "empty_input", "detail": str(exc)})


@app.exception_handler(PolicyNotFoundError)
async def handle_policy_not_found(request: Request, exc: PolicyNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "policy_not_found", "detail": str(exc)})


@app.exception_handler(JudgeRefusalError)
async def handle_judge_refusal(request: Request, exc: JudgeRefusalError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "judge_refusal", "detail": str(exc)})


@app.exception_handler(JudgeSchemaViolationError)
async def handle_judge_schema_violation(request: Request, exc: JudgeSchemaViolationError) -> JSONResponse:
    logger.warning("judge output failed schema validation after local retries: %s", exc)
    return JSONResponse(status_code=422, content={"error": "judge_schema_violation", "detail": str(exc)})


@app.exception_handler(JudgeTruncationError)
async def handle_judge_truncation(request: Request, exc: JudgeTruncationError) -> JSONResponse:
    logger.error("judge truncation (raise max_tokens): %s", exc)
    return JSONResponse(status_code=500, content={"error": "judge_truncation", "detail": str(exc)})


@app.exception_handler(OpenAIError)
async def handle_openai_error(request, exc: OpenAIError):
    logger.error(f"Upstream OpenAI/OpenRouter Fault: {str(exc)}")
    
    return JSONResponse(
        status_code=502,
        content={
            "error": "upstream_unavailable",
            "detail": "The compliance judge model is temporarily experiencing network latency or availability limits. Please retry your request shortly."
        }
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "an unexpected error occurred"})


# --- routes ---


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)  
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    configured = bool(settings.openai_api_key.get_secret_value())
    return HealthResponse(
        status="ok" if configured else "degraded",
        timestamp=datetime.utcnow(),
        openai_configured=configured,
    )


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate(
    payload: EvaluateRequest, evaluator: PolicyEvaluator = Depends(get_evaluator)
) -> EvaluationResult:
    return await evaluator.evaluate(payload.business_description, payload.policy_id, payload.locale)


@app.post("/batch", response_model=BatchEvaluateResponse)
async def batch_evaluate(
    payload: BatchEvaluateRequest, evaluator: PolicyEvaluator = Depends(get_evaluator)
) -> BatchEvaluateResponse:
    items = [(item.business_description, item.policy_id, item.locale) for item in payload.items]
    raw_results = await evaluator.evaluate_batch(items)

    results = [
        BatchItemResult(index=r.index, status="failure", error=r.error)
        if isinstance(r, EvaluationFailure)
        else BatchItemResult(index=i, status="success", result=r)
        for i, r in enumerate(raw_results)
    ]
    return BatchEvaluateResponse(results=results)