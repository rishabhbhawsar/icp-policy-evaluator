"""
src/models/taxonomy.py

Core data contracts for the ICP Taxonomy Policy Evaluation Framework.

Design intent
-------------
These models define the *shape* every policy, locale, and LLM judgment must
conform to. They deliberately do NOT hardcode all 53 taxonomy policies as
Python identifiers -- taxonomy content is data (loaded from a JSON/DB-backed
policy registry at runtime), not code. Enumerating every policy as an Enum
member would mean a regulatory update requires a code deploy, which is the
wrong coupling for a compliance system.

What IS hardcoded here, as closed enums, is the small set of things that are
structurally stable and where an unrecognized value should be a loud
validation failure rather than a silent fallback: the 17 supported locales
and the ~10 top-level policy categories the 53 leaf policies are grouped
under.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SupportedLocale(str, Enum):
    """ISO 3166-1 alpha-2 codes for the 17 markets this taxonomy covers.

    A closed enum is deliberate: a request for a locale outside this set
    must fail loudly at the API boundary (HTTP 422) instead of silently
    falling through to some default policy -- in a compliance context,
    "we don't have a rule for this" and "this is definitely fine" must
    never look the same to a caller.
    """

    US = "US"
    GB = "GB"
    DE = "DE"
    FR = "FR"
    ES = "ES"
    IT = "IT"
    NL = "NL"
    IE = "IE"
    SG = "SG"
    AU = "AU"
    JP = "JP"
    IN = "IN"
    BR = "BR"
    MX = "MX"
    CA = "CA"
    AE = "AE"
    ZA = "ZA"


class PolicyCategory(str, Enum):
    """Stable top-level product domain categories.

    The 53 individual leaf policies live in the policy registry (data),
    each tagged with one of these categories. Categories move on the order
    of years; leaf policies move on the order of weeks -- hence the split
    between "hardcoded enum" and "loaded data."
    """

    LENDING = "LENDING"
    PAYMENTS = "PAYMENTS"
    REMITTANCE = "REMITTANCE"
    BNPL = "BNPL"
    INVESTMENTS = "INVESTMENTS"
    INSURANCE = "INSURANCE"
    CRYPTO_DIGITAL_ASSETS = "CRYPTO_DIGITAL_ASSETS"
    MERCHANT_SERVICES = "MERCHANT_SERVICES"
    KYC_AML = "KYC_AML"
    DATA_PRIVACY = "DATA_PRIVACY"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    PROHIBITED = "PROHIBITED"


_RULE_ID_PATTERN = r"^[A-Z]{2,10}-\d{3}$"  # e.g. "KYC-014"


class ComplianceRule(BaseModel):
    """A single, atomic, checkable rule inside a policy.

    Kept deliberately granular -- one rule = one checkable condition -- so
    an LLM judge's violation list can point at *exactly* which rule fired,
    instead of a vague "this failed policy X" verdict a human auditor then
    has to reverse-engineer from the raw text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(
        ...,
        pattern=_RULE_ID_PATTERN,
        description="e.g. 'KYC-014'. Enforced pattern keeps rule IDs greppable across the ledger.",
    )
    description: str = Field(..., min_length=1, max_length=500)
    risk_level: RiskLevel
    required_disclosures: list[str] = Field(default_factory=list)
    prohibited_terms: list[str] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def _no_blank_description(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("description cannot be blank/whitespace-only")
        return v


class ICPTaxonomy(BaseModel):
    """A single locale-scoped policy record: the unit the evaluator judges against."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(..., pattern=_RULE_ID_PATTERN)
    category: PolicyCategory
    locale: SupportedLocale
    version: int = Field(
        ..., ge=1, description="Monotonic version. Bump on any rule change; never mutate a version in place."
    )
    effective_date: date
    rules: list[ComplianceRule] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _prohibited_rules_have_no_disclosure_escape_hatch(self) -> "ICPTaxonomy":
        """A PROHIBITED-risk rule cannot also define required_disclosures.

        'Prohibited' means no amount of disclosure makes the item
        compliant -- if a rule tries to be both, that's a contradiction in
        the source data itself, and we want that surfaced at policy-load
        time, not discovered later when an LLM judgment looks inconsistent.
        """
        for rule in self.rules:
            if rule.risk_level == RiskLevel.PROHIBITED and rule.required_disclosures:
                raise ValueError(
                    f"Rule {rule.rule_id} is PROHIBITED but defines required_disclosures; "
                    "prohibited rules cannot be disclosed into compliance."
                )
        return self


class Classification(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class EvaluationResult(BaseModel):
    """
    The exact structural contract we force the LLM judge to emit via JSON mode.

    This is the single most important schema in the project: it's the
    boundary between "text a model produced" and "data the rest of the
    system is allowed to trust." Every field exists to make one specific
    failure mode visible instead of silently swallowed.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    evaluation_id: UUID = Field(default_factory=uuid4)
    policy_id: str = Field(..., pattern=_RULE_ID_PATTERN)
    locale: SupportedLocale
    classification: Classification
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    violated_rule_ids: list[str] = Field(default_factory=list)
    reasoning: str = Field(
        ..., min_length=1, max_length=1000, description="Capped to keep the ledger bounded and the judge concise."
    )
    model_name: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    cache_hit: bool = False

    @field_validator("violated_rule_ids")
    @classmethod
    def _dedupe_and_sort_violations(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _risk_level_consistent_with_classification(self) -> "EvaluationResult":
        """Risk level must match the semantic meaning of the classification.

        COMPLIANT means the item passed every rule -- residual risk is LOW by
        definition. NON_COMPLIANT means at least one rule was violated --
        that's HIGH or PROHIBITED risk. REQUIRES_HUMAN_REVIEW sits in between.

        Without this, the judge can emit COMPLIANT + HIGH, which is internally
        contradictory and useless to a downstream auditor.
        """
        if (
            self.classification == Classification.COMPLIANT
            and self.risk_level != RiskLevel.LOW
        ):
            raise ValueError(
                "COMPLIANT classification requires risk_level=LOW "
                f"(got {self.risk_level.value})"
            )
        if (
            self.classification == Classification.NON_COMPLIANT
            and self.risk_level not in (RiskLevel.HIGH, RiskLevel.PROHIBITED)
        ):
            raise ValueError(
                "NON_COMPLIANT classification requires risk_level=HIGH or PROHIBITED "
                f"(got {self.risk_level.value})"
            )
        if (
            self.classification == Classification.REQUIRES_HUMAN_REVIEW
            and self.risk_level != RiskLevel.MEDIUM
        ):
            raise ValueError(
                "REQUIRES_HUMAN_REVIEW classification requires risk_level=MEDIUM "
                f"(got {self.risk_level.value})"
            )
        return self