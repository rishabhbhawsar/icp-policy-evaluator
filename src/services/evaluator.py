"""src/services/evaluator.py

Orchestration layer: validate input -> cache lookup -> build locale-scoped
prompt -> dispatch to judge -> write ledger -> write cache. Batch path adds
semaphore-bounded asyncio.gather so one slow/failed item doesn't block or
crash the rest.

Cache, ledger, and taxonomy storage are Protocols, not concrete classes:
this file has no opinion on SQLite vs Redis vs anything else. That binding
happens at composition time (main.py / config.py), not here.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAIError

from src.models.taxonomy import EvaluationResult, ICPTaxonomy, SupportedLocale
from src.services.openai_client import JudgeRefusalError, JudgeTruncationError, OpenAIJudgeClient

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24h; the policy version is already part of the cache key,
# so a rule change invalidates via key change, not via TTL expiry. TTL only bounds cache growth.
DEFAULT_BATCH_CONCURRENCY = 10


class TaxonomyRepository(Protocol):
    async def get(self, policy_id: str, locale: SupportedLocale) -> ICPTaxonomy | None: ...


class CacheBackend(Protocol):
    async def get(self, key: str) -> EvaluationResult | None: ...
    async def set(self, key: str, value: EvaluationResult, ttl_seconds: int) -> None: ...


class LedgerWriter(Protocol):
    async def record(self, *, cache_key: str, business_description: str, result: EvaluationResult) -> None: ...


class EmptyInputError(ValueError):
    """business_description is empty or whitespace-only."""


class PolicyNotFoundError(LookupError):
    """No ICPTaxonomy record for (policy_id, locale)."""


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    index: int
    policy_id: str
    locale: SupportedLocale
    error: str


class PolicyEvaluator:
    def __init__(
        self,
        judge_client: OpenAIJudgeClient,
        taxonomy_repo: TaxonomyRepository,
        cache: CacheBackend,
        ledger: LedgerWriter,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    ) -> None:
        self._client = judge_client
        self._taxonomy_repo = taxonomy_repo
        self._cache = cache
        self._ledger = ledger
        self._cache_ttl_seconds = cache_ttl_seconds
        self._semaphore = asyncio.Semaphore(batch_concurrency)

    @staticmethod
    def _validate_input(business_description: str) -> str:
        stripped = business_description.strip()
        if not stripped:
            raise EmptyInputError("business_description is empty or whitespace-only")
        return stripped

    @staticmethod
    def _cache_key(business_description: str, policy: ICPTaxonomy) -> str:
        # policy_id + locale + version + description: a rule-content change bumps
        # `version`, which changes the key -- no need to invalidate cache entries by hand.
        digest_input = f"{policy.policy_id}:{policy.locale.value}:{policy.version}:{business_description}"
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_system_prompt(policy: ICPTaxonomy) -> str:
        rule_lines = [
            f"- [{rule.rule_id}] risk={rule.risk_level.value} :: {rule.description} "
            f"(required_disclosures: {'; '.join(rule.required_disclosures) or 'none'}; "
            f"prohibited_terms: {'; '.join(rule.prohibited_terms) or 'none'})"
            for rule in policy.rules
        ]
        return (
            f"You are a compliance judge for policy {policy.policy_id} "
            f"(category={policy.category.value}, locale={policy.locale.value}, version={policy.version}).\n"
            "Evaluate the business description strictly against these rules:\n"
            + "\n".join(rule_lines)
            + "\nCite every violated rule_id explicitly. Do not invent rules outside this list. "
            "If the description lacks enough information to judge with confidence, classify as "
            "REQUIRES_HUMAN_REVIEW rather than guessing."
        )

    async def evaluate(
        self, business_description: str, policy_id: str, locale: SupportedLocale
    ) -> EvaluationResult:
        description = self._validate_input(business_description)

        policy = await self._taxonomy_repo.get(policy_id, locale)
        if policy is None:
            raise PolicyNotFoundError(f"no policy {policy_id} for locale {locale.value}")

        cache_key = self._cache_key(description, policy)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        system_prompt = self._build_system_prompt(policy)
        result = await self._client.evaluate(system_prompt=system_prompt, user_prompt=description)

        await self._ledger.record(cache_key=cache_key, business_description=description, result=result)
        await self._cache.set(cache_key, result, self._cache_ttl_seconds)
        return result

    async def _evaluate_bounded(
        self, index: int, business_description: str, policy_id: str, locale: SupportedLocale
    ) -> EvaluationResult | EvaluationFailure:
        async with self._semaphore:
            try:
                return await self.evaluate(business_description, policy_id, locale)
            except (EmptyInputError, PolicyNotFoundError, JudgeRefusalError, JudgeTruncationError, OpenAIError) as exc:
                logger.warning(
                    "batch item failed index=%d policy_id=%s locale=%s: %s",
                    index, policy_id, locale.value, exc,
                )
                return EvaluationFailure(index=index, policy_id=policy_id, locale=locale, error=str(exc))

    async def evaluate_batch(
        self, items: list[tuple[str, str, SupportedLocale]]
    ) -> list[EvaluationResult | EvaluationFailure]:
        """items: (business_description, policy_id, locale) tuples. Order-preserving.
        Semaphore bounds concurrent in-flight requests; failures are returned as typed
        EvaluationFailure entries rather than raised, so one bad item never cancels
        the rest of the batch."""
        tasks = [self._evaluate_bounded(i, desc, pid, loc) for i, (desc, pid, loc) in enumerate(items)]
        return await asyncio.gather(*tasks)