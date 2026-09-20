"""tests/test_harness.py

Manual integration smoke test. Instantiates its own OpenAIJudgeClient and
PolicyEvaluator, reads config via get_settings(), and writes through to the
real SQLite ledger/cache at settings.database_path. Not a pytest suite --
run directly:

    python -m tests.test_harness
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date

from openai import OpenAIError

from src.core.config import get_settings
from src.models.taxonomy import (
    ComplianceRule,
    ICPTaxonomy,
    PolicyCategory,
    RiskLevel,
    SupportedLocale,
)
from src.services.evaluator import EmptyInputError, PolicyEvaluator, PolicyNotFoundError
from src.services.openai_client import JudgeRefusalError, JudgeTruncationError, OpenAIJudgeClient
from src.services.persistence import SQLiteCacheBackend, SQLiteLedgerWriter

_POLICY_ID = "KYC-014"
_LOCALE = SupportedLocale.US

_SEED_POLICY = ICPTaxonomy(
    policy_id=_POLICY_ID,
    category=PolicyCategory.KYC_AML,
    locale=_LOCALE,
    version=1,
    effective_date=date(2026, 1, 1),
    rules=[
        ComplianceRule(
            rule_id=_POLICY_ID,
            description="Beneficial ownership must be disclosed for entities holding >25% equity.",
            risk_level=RiskLevel.HIGH,
            required_disclosures=["beneficial_owner_identity"],
        )
    ],
)


class _StaticTaxonomyRepo:
    """Single-policy stand-in. No SQLite-backed policy registry exists yet
    (see main.py); this harness needs one known-good policy, not a full repo."""

    async def get(self, policy_id: str, locale: SupportedLocale) -> ICPTaxonomy | None:
        return _SEED_POLICY if (policy_id, locale) == (_POLICY_ID, _LOCALE) else None


# label, description, expected note (for terminal output only -- not asserted programmatically)
_TEST_CASES = [
    (
        "clear_compliant",
        "A small-business lending platform. Sole owner Jane Doe holds 100% equity; "
        "beneficial ownership has been disclosed to regulators and is documented in "
        "onboarding records.",
        "expect COMPLIANT",
    ),
    (
        "ambiguous_missing_disclosure",
        "A lending business that offers short-term loans to small business owners.",
        "expect NON_COMPLIANT or REQUIRES_HUMAN_REVIEW -- no ownership disclosure stated",
    ),
    (
        "empty_input",
        "   ",
        "expect EmptyInputError -- rejected before any judge call",
    ),
]


def _count_ledger_rows(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM compliance_ledger").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def _print_outcome(label: str, description: str, note: str, *, result=None, error: str | None = None) -> None:
    print(f"\n[{label}] ({note})")
    print(f"  input          : {description!r}")
    if error is not None:
        print(f"  outcome        : REJECTED -- {error}")
        return
    print(f"  classification : {result.classification.value}")
    print(f"  confidence     : {result.confidence:.2f}")
    print(f"  risk_level     : {result.risk_level.value}")
    print(f"  violated_rules : {result.violated_rule_ids}")
    print(f"  reasoning      : {result.reasoning}")


async def main() -> None:
    settings = get_settings()

    judge_client = OpenAIJudgeClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )
    ledger = await SQLiteLedgerWriter.create(settings.database_path)
    cache = await SQLiteCacheBackend.create(settings.database_path)
    evaluator = PolicyEvaluator(
        judge_client=judge_client,
        taxonomy_repo=_StaticTaxonomyRepo(),
        cache=cache,
        ledger=ledger,
        cache_ttl_seconds=settings.cache_ttl_seconds,
        batch_concurrency=settings.batch_concurrency_limit,
    )

    ledger_rows_before = _count_ledger_rows(settings.database_path)

    for label, description, note in _TEST_CASES:
        try:
            result = await evaluator.evaluate(description, _POLICY_ID, _LOCALE)
            _print_outcome(label, description, note, result=result)
        except (EmptyInputError, PolicyNotFoundError, JudgeRefusalError, JudgeTruncationError, OpenAIError) as exc:
            _print_outcome(label, description, note, error=str(exc))

    await judge_client.aclose()
    await cache.close()
    await ledger.close()

    ledger_rows_after = _count_ledger_rows(settings.database_path)
    written = ledger_rows_after - ledger_rows_before

    print(f"\n[ledger audit] db path              : {settings.database_path}")
    print(f"[ledger audit] compliance_ledger rows written this run : {written}")
    print("[ledger audit] expected 2 -- clear_compliant + ambiguous_missing_disclosure; "
          "empty_input never reaches the judge or the ledger by design")


if __name__ == "__main__":
    asyncio.run(main())