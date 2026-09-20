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
    # --- Shell structures: phrasing deliberately varied per case so there is no
    # single repeated giveaway phrase correlated with the label ---
    LabeledCase(
        "shell_layered_but_fully_disclosed",
        "Meridian Capital Holdings Ltd. operates a three-tier structure: the "
        "Cayman parent wholly owns a BVI trading entity, which wholly owns the "
        "US operating company. Alistair Chen controls the full equity stack, "
        "and his identity appears in each tier's regulatory filing, "
        "cross-referenced against passport records collected at onboarding.",
        expected_compliant=True,
    ),
    LabeledCase(
        "shell_blended_svp_undisclosed",
        "Solstice Global Partners is a joint venture between three regional "
        "holding vehicles feeding into a blended special-purpose vehicle. The "
        "partnership deed submitted with this application leaves the schedule "
        "for each partner's percentage stake blank, pending a later amendment.",
        expected_compliant=False,
    ),
    LabeledCase(
        "shell_trust_fully_disclosed",
        "Northbridge Trust Company administers an irrevocable trust holding "
        "the entire equity of Northbridge Lending Solutions LLC. The trust "
        "deed submitted at onboarding names Eleanor Voss as the sole party "
        "entitled to trust income and corpus, with no other beneficiaries "
        "listed.",
        expected_compliant=True,
    ),
    LabeledCase(
        "shell_nominee_directors_deferred",
        "Cascade Ventures Group's incorporation paperwork across four "
        "jurisdictions lists only professional nominee directors; none of the "
        "underlying principals who actually hold equity appear anywhere in "
        "the documents submitted with this application.",
        expected_compliant=False,
    ),
    LabeledCase(
        "shell_family_trust_fully_disclosed",
        "Ironwood Family Office Holdings' beneficiary schedule, filed "
        "alongside its onboarding packet, names a single individual as "
        "entitled to the entirety of distributions across all three "
        "affiliated lending subsidiaries.",
        expected_compliant=True,
    ),
    LabeledCase(
        "shell_rotating_allocation_undisclosed",
        "Aurelian Global Structures reassigns equity stakes among affiliated "
        "principals under a rolling capital allocation strategy. The "
        "allocation memorandum governing who currently holds what stake is "
        "referenced in the application but was not attached, and the version "
        "on file is eleven months out of date.",
        expected_compliant=False,
    ),
    # --- Ambiguous fintech/neobanking: ownership simply never comes up.
    # Expect REQUIRES_HUMAN_REVIEW, not a confident denial. ---
    LabeledCase(
        "neobank_throughput_only",
        "Flux Neobank operates a cross-border payment corridor connecting SME "
        "merchants across Southeast Asia and Europe, processing upwards of "
        "40,000 transactions daily. Its application materials focus entirely "
        "on throughput, uptime SLAs, and settlement latency benchmarks.",
        expected_compliant=False,
    ),
    LabeledCase(
        "neobank_marketing_only",
        "Zenith Pay is a challenger banking platform offering multi-currency "
        "wallets and instant remittance for freelancers, emphasizing its FX "
        "spread and 99.98% platform uptime as its core pitch to prospective "
        "partners.",
        expected_compliant=False,
    ),
    LabeledCase(
        "neobank_wrong_certifications",
        "Orbit Financial Technologies provides an embedded-finance API layer "
        "for e-commerce platforms, highlighting its SOC 2 Type II and "
        "PCI-DSS Level 1 certifications as its primary trust signals for "
        "enterprise partners.",
        expected_compliant=False,
    ),
    LabeledCase(
        "neobank_partnerships_only",
        "Driftwood Remit facilitates high-frequency micro-remittances for "
        "migrant worker communities, citing partnerships with four regional "
        "payment networks and a goal of cutting average fees below 1.5% by "
        "Q3.",
        expected_compliant=False,
    ),
    LabeledCase(
        "neobank_infra_only",
        "Cobalt Card Systems provides white-label prepaid card issuance "
        "infrastructure for fintech startups, emphasizing rapid BIN "
        "sponsorship onboarding and a self-service developer dashboard.",
        expected_compliant=False,
    ),
    # --- Explicit decliners: each states outright, in its own words, that it
    # does not identify who controls the entity. Expect a confident denial. ---
    LabeledCase(
        "decliner_privacy_first",
        "Umbra Pay's onboarding flow never asks a merchant to name a "
        "controlling shareholder -- no such field exists anywhere in its "
        "account-creation schema, a design choice the company describes as "
        "core to its privacy-first architecture.",
        expected_compliant=False,
    ),
    LabeledCase(
        "decliner_anonymous_p2p",
        "Nomad Remit Networks lets any node operator register under a wallet "
        "address alone. The protocol has no mechanism for attaching a "
        "real-world identity to that address, and the team has stated no "
        "plans to add one.",
        expected_compliant=False,
    ),
    LabeledCase(
        "decliner_intentionally_opaque",
        "Vantablack Financial's submitted paperwork lists its principals only "
        "as 'Managing Member A' and 'Managing Member B' throughout every "
        "document, with counsel stating that further identification 'is not "
        "something the firm provides.'",
        expected_compliant=False,
    ),
    LabeledCase(
        "decliner_zero_knowledge",
        "Ghostwire Settlements' network is architected so that no party, "
        "including the operator itself, can determine who ultimately "
        "controls any given settlement node.",
        expected_compliant=False,
    ),
    # --- Surface-risky but genuinely compliant: tests whether the judge scores
    # actual disclosure adequacy or reacts to risk-coded business descriptions ---
    LabeledCase(
        "risky_surface_trade_finance_disclosed",
        "Obsidian Trade Finance Group underwrites high-risk trade finance "
        "instruments for commodity traders in volatile emerging markets. Its "
        "shareholder register, submitted with this application, lists one "
        "individual as holding all outstanding shares, a fact cross-checked "
        "against the firm's annual regulatory filings.",
        expected_compliant=True,
    ),
    LabeledCase(
        "risky_surface_offshore_derivatives_disclosed",
        "Meridian Offshore Capital Markets structures complex leveraged "
        "derivatives through a Cayman special-purpose vehicle for "
        "institutional counterparties. Despite the offshore structure, the "
        "SPV's formation documents name a single individual as its sole "
        "member, notarized and filed with this application.",
        expected_compliant=True,
    ),
    LabeledCase(
        "risky_surface_distressed_lending_disclosed",
        "Tempest Cross-Border Lending extends high-yield bridge financing to "
        "distressed borrowers in politically volatile markets. Its annual "
        "filings have named the same individual as sole shareholder every "
        "year since incorporation, a continuity confirmed against public "
        "companies-registry records.",
        expected_compliant=True,
    ),
    LabeledCase(
        "risky_surface_npl_securitization_disclosed",
        "Blackridge Structured Finance issues asset-backed securities "
        "collateralized by non-performing loan portfolios bought at deep "
        "discounts. Ownership of the issuing entity traces to a single named "
        "individual, confirmed via the notarized incorporation certificate "
        "attached to this application.",
        expected_compliant=True,
    ),
    LabeledCase(
        "risky_surface_frontier_credit_disclosed",
        "Palisade Emerging Markets Credit originates unsecured consumer "
        "credit in high-default-rate frontier markets with limited credit "
        "bureau infrastructure. Its cap table, refreshed annually and "
        "attached here, has listed the same sole shareholder since the "
        "company's founding.",
        expected_compliant=True,
    ),
    # --- Trap cases: surface cue and correct answer point in opposite
    # directions. A keyword-matcher fails these even where it passes everything
    # above; a judge reasoning about the actual >25% threshold should not. ---
    LabeledCase(
        "trap_no_individual_above_threshold",
        "Sterling Cross Capital's onboarding packet names its parent, "
        "Sterling Holdings PLC, as sole shareholder of record. Sterling "
        "Holdings PLC is itself a publicly traded company with no single "
        "shareholder controlling more than 4% of outstanding stock.",
        expected_compliant=True,
    ),
    LabeledCase(
        "trap_named_minority_unnamed_majority",
        "Northfield Business Solutions' onboarding file names Robert Guerra "
        "as holding a 20% equity stake and lists no other shareholders by "
        "name, though registry records referenced in the application "
        "indicate the remaining 80% is held via an unnamed 'affiliated "
        "investment vehicle.'",
        expected_compliant=False,
    ),
    LabeledCase(
        "trap_disclosure_claimed_elsewhere",
        "Harrow & Vance Underwriting states that its majority owner 'prefers "
        "not to be named in commercial documentation' but confirms that full "
        "ownership records, including notarized identity verification, have "
        "been lodged directly with the national financial regulator, outside "
        "the materials submitted with this application.",
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