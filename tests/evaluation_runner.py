"""tests/evaluation_runner.py

Benchmark harness: runs a labeled test matrix through the real
PolicyEvaluator pipeline (real judge calls, real ledger/cache) and computes
precision/recall/F1/accuracy from a confusion matrix. Run directly:

    python -m tests.evaluation_runner

Provider config comes from get_settings() -- no hardcoded base_url/model
here; that value lives in .env via src/core/config.py, not retyped per file.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from openai import OpenAIError

from src.core.config import get_settings
from src.models.taxonomy import Classification
from src.services.evaluator import (
    EmptyInputError,
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
from tests.test_harness import _LOCALE, _POLICY_ID, _StaticTaxonomyRepo

_EPSILON = 1e-9

_JUDGE_EXCEPTIONS = (
    EmptyInputError,
    PolicyNotFoundError,
    JudgeRefusalError,
    JudgeTruncationError,
    JudgeSchemaViolationError,
    OpenAIError,
)


@dataclass(frozen=True, slots=True)
class LabeledCase:
    label: str
    description: str
    expected_compliant: bool


_TEST_MATRIX: list[LabeledCase] = [
    LabeledCase(
        "standard_full_disclosure",
        "A small-business lending platform. Sole owner Maria Alonso holds 100% "
        "equity; beneficial ownership has been disclosed to regulators and is "
        "documented in onboarding records.",
        expected_compliant=True,
    ),
    LabeledCase(
        "complex_offshore_but_verified",
        "A multi-layered corporate holding company with offshore subsidiaries "
        "across three jurisdictions. All beneficial owners at every layer have "
        "been identified, verified, and fully disclosed to regulators per "
        "KYC-014, with notarized ownership charts on file.",
        expected_compliant=True,
    ),
    LabeledCase(
        "missing_disclosure",
        "A small business offering short-term commercial loans to local retailers.",
        expected_compliant=False,
    ),
    LabeledCase(
        "explicit_non_tracking",
        "A high-concurrency remittance processor handling cross-border transfers. "
        "The platform explicitly does not maintain beneficial ownership records "
        "or any regulatory tracking structures for its corporate clients.",
        expected_compliant=False,
    ),
]


@dataclass
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    errors: list[str] = field(default_factory=list)
    deferred_negatives: int = 0  # REQUIRES_HUMAN_REVIEW bucketed as negative per spec; tracked separately below

    @property
    def n(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + _EPSILON)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + _EPSILON)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r + _EPSILON)

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.n + _EPSILON)


def _classify(classification: Classification, expected_compliant: bool) -> tuple[str, bool]:
    """Returns (bucket, is_deferred). REQUIRES_HUMAN_REVIEW is bucketed with
    NON_COMPLIANT as a conservative "not approved" outcome per spec, but
    is_deferred is tracked so an escalation-to-human isn't silently counted
    with the same confidence as an explicit NON_COMPLIANT denial."""
    predicted_compliant = classification == Classification.COMPLIANT
    is_deferred = classification == Classification.REQUIRES_HUMAN_REVIEW

    if predicted_compliant and expected_compliant:
        return "tp", is_deferred
    if not predicted_compliant and not expected_compliant:
        return "tn", is_deferred
    if predicted_compliant and not expected_compliant:
        return "fp", is_deferred
    return "fn", is_deferred


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

    matrix = ConfusionMatrix()
    rows: list[tuple[str, bool, str, str]] = []

    for case in _TEST_MATRIX:
        try:
            result = await evaluator.evaluate(case.description, _POLICY_ID, _LOCALE)
        except _JUDGE_EXCEPTIONS as exc:
            matrix.errors.append(f"{case.label}: {exc}")
            rows.append((case.label, case.expected_compliant, "ERROR", "excluded"))
            continue

        bucket, is_deferred = _classify(result.classification, case.expected_compliant)
        if bucket == "tp":
            matrix.tp += 1
        elif bucket == "tn":
            matrix.tn += 1
        elif bucket == "fp":
            matrix.fp += 1
        else:
            matrix.fn += 1
        if is_deferred and bucket in ("tn", "fn"):
            matrix.deferred_negatives += 1

        rows.append((case.label, case.expected_compliant, result.classification.value, bucket))

    await judge_client.aclose()
    await cache.close()
    await ledger.close()

    _print_report(rows, matrix)


def _print_report(rows: list[tuple[str, bool, str, str]], matrix: ConfusionMatrix) -> None:
    print("\n" + "=" * 64)
    print("EVALUATION RUN -- per-case results")
    print("=" * 64)
    for label, expected, predicted, bucket in rows:
        print(f"  {label:<28} expected={str(expected):<5} predicted={predicted:<20} -> {bucket.upper()}")

    if matrix.errors:
        print("\n  Excluded from matrix (judge call failed, not a classification):")
        for err in matrix.errors:
            print(f"    - {err}")

    print("\n" + "=" * 64)
    print("CONFUSION MATRIX")
    print("=" * 64)
    print(f"  TP: {matrix.tp}   TN: {matrix.tn}   FP: {matrix.fp}   FN: {matrix.fn}   (n={matrix.n})")
    if matrix.deferred_negatives:
        print(
            f"  Note: {matrix.deferred_negatives} of the negative bucket(s) above were "
            "REQUIRES_HUMAN_REVIEW, not an explicit NON_COMPLIANT denial -- bucketed as "
            "negative per spec, flagged here rather than silently merged."
        )

    print("\n" + "=" * 64)
    print("METRICS")
    print("=" * 64)
    print(f"  Precision : {matrix.precision:.3f}")
    print(f"  Recall    : {matrix.recall:.3f}")
    print(f"  F1-Score  : {matrix.f1:.3f}")
    print(f"  Accuracy  : {matrix.accuracy:.3f}")
    print("=" * 64)

    if matrix.n:
        swing = 100 / matrix.n
        print(
            f"\nNOTE: n={matrix.n}. One flipped classification moves accuracy by "
            f"~{swing:.0f} percentage points at this sample size. These numbers prove "
            "the harness's mechanics are correct, not a statistically defensible claim -- "
            "treat any figure above as resume-ready only after running against a labeled "
            "set large enough (dozens+ per class) for the confidence interval to be tight."
        )


if __name__ == "__main__":
    asyncio.run(main())