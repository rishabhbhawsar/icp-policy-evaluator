from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SupportedLocale(str, Enum):
    """Supported market configurations mapped to ISO 3166-1 alpha-2 boundaries."""
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
    """Top-level domains used for indexing dynamic leaf policies."""
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


_RULE_ID_PATTERN = r"^[A-Z]{2,10}-\d{3}$"


class ComplianceRule(BaseModel):
    """Data contract representing an atomic compliance constraint rule."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(..., pattern=_RULE_ID_PATTERN)
    description: str = Field(..., min_length=1, max_length=500)
    risk_level: RiskLevel
    required_disclosures: list[str] = Field(default_factory=list)
    prohibited_terms: list[str] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def _validate_description_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Description cannot be empty or whitespace only.")
        return v


class ICPTaxonomy(BaseModel):
    """Target schema configuration for a locale-scoped dynamic compliance rulebook."""
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(..., pattern=_RULE_ID_PATTERN)
    category: PolicyCategory
    locale: SupportedLocale
    version: int = Field(..., ge=1)
    effective_date: date
    rules: list[ComplianceRule] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_prohibited_constraints(self) -> ICPTaxonomy:
        for rule in self.rules:
            if rule.risk_level == RiskLevel.PROHIBITED and rule.required_disclosures:
                raise ValueError(
                    f"Invalid Policy: Rule {rule.rule_id} cannot define required disclosures while marked PROHIBITED."
                )
        return self


class Classification(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class EvaluationResult(BaseModel):
    """Strict structural schema schema contract forced onto LLM Judge outputs."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    evaluation_id: UUID = Field(default_factory=uuid4)
    policy_id: str = Field(..., pattern=_RULE_ID_PATTERN)
    locale: SupportedLocale
    classification: Classification
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    violated_rule_ids: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., min_length=1, max_length=1000)
    model_name: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("violated_rule_ids")
    @classmethod
    def _dedupe_and_sort_violations(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _validate_verdict_consistency(self) -> EvaluationResult:
        if self.classification == Classification.COMPLIANT and self.violated_rule_ids:
            raise ValueError("COMPLIANT classification cannot contain violated rule IDs.")
        if self.classification == Classification.NON_COMPLIANT and not self.violated_rule_ids:
            raise ValueError("NON_COMPLIANT classification must cite at least one explicit violated rule ID.")
        return self
