"""src/services/openai_client.py

Async client wrapper providing structured EvaluationResult output over any
OpenAI-compatible chat completions endpoint (OpenAI, Groq, and similar), via
a configurable base_url. AsyncOpenAI raises openai.* exception types based on
HTTP status/transport behavior, not on which server answered -- the retry
policy below is therefore correct regardless of provider.

Not every OpenAI-compatible provider supports OpenAI's proprietary strict-
schema mode (.chat.completions.parse() / response_format={"type":
"json_schema", "strict": true}); Groq, for example, restricts it to a short
model allow-list. This client instead uses the widely-supported
response_format={"type": "json_object"} (valid-JSON guarantee, no schema
guarantee) and validates the result against a narrow judgment schema itself,
with a bounded local retry for the case where a provider's best-effort JSON
doesn't match on the first attempt. Identity/audit fields (policy_id,
locale, model_name) are populated from the caller and the client's own
config, never trusted to the model's self-report.
"""

from __future__ import annotations

import json
import logging
import re

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.models.taxonomy import Classification, EvaluationResult, RiskLevel, SupportedLocale

logger = logging.getLogger("uvicorn.error")

RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

DEFAULT_MODEL = "gpt-4o-2024-08-06"
_MAX_SCHEMA_ATTEMPTS = 2  # local retries for "valid JSON, wrong shape" -- separate from tenacity's transient-error retry
_JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_OUTPUT_FORMAT_INSTRUCTIONS = (
    "\n\nRespond with ONLY a single JSON object with exactly these fields, no others:\n"
    '{"classification": "COMPLIANT" | "NON_COMPLIANT" | "REQUIRES_HUMAN_REVIEW", '
    '"confidence": <float 0.0-1.0>, '
    '"risk_level": "LOW" | "MEDIUM" | "HIGH" | "PROHIBITED", '
    '"violated_rule_ids": [<rule id strings>, empty list if none], '
    '"reasoning": "<concise explanation, under 1000 characters>"}\n'
    "No prose outside the JSON object. No markdown code fences."
)


class JudgeRefusalError(RuntimeError):
    """Model declined, content-filtered, or returned an empty body. Not retryable."""


class JudgeTruncationError(RuntimeError):
    """Response cut off at max_tokens before it could complete. Not retryable
    without raising the cap."""


class JudgeSchemaViolationError(RuntimeError):
    """Provider returned syntactically valid JSON that failed judgment-schema
    validation on every local retry. Distinct from truncation: the response
    completed, it just didn't match the required shape -- this is the real,
    measurable schema-adherence failure mode for non-strict-mode providers."""


class _JudgmentPayload(BaseModel):
    """The only fields the model is asked to produce. Narrower than
    EvaluationResult on purpose: policy_id, locale, model_name,
    evaluation_id, schema_version, and evaluated_at are already known to the
    caller and are filled in by OpenAIJudgeClient.evaluate(), not by the LLM."""

    model_config = ConfigDict(extra="forbid")

    classification: Classification
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    violated_rule_ids: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., min_length=1, max_length=1000)


def _strip_json_fences(raw: str) -> str:
    return _JSON_FENCE_PATTERN.sub("", raw.strip())


class OpenAIJudgeClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_tokens: int = 800,
    ) -> None:
        # base_url=None keeps the SDK's own default (api.openai.com). Any other
        # OpenAI-compatible endpoint (e.g. "https://api.groq.com/openai/v1") is
        # a drop-in swap -- the request/response shape and every exception type
        # below is unchanged by which server answers.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._model = model
        self._max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _create(self, *, system_prompt: str, user_prompt: str):
        return await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt + _OUTPUT_FORMAT_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=self._max_tokens,
            temperature=0,
        )

    async def evaluate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        policy_id: str,
        locale: SupportedLocale,
    ) -> EvaluationResult:
        last_error: Exception | None = None
        prompt = system_prompt

        for attempt in range(1, _MAX_SCHEMA_ATTEMPTS + 1):
            completion = await self._create(system_prompt=prompt, user_prompt=user_prompt)
            choice = completion.choices[0]

            if choice.finish_reason == "content_filter" or choice.message.refusal:
                raise JudgeRefusalError(choice.message.refusal or "provider content filter blocked the response")

            if choice.finish_reason == "length":
                raise JudgeTruncationError(f"response truncated at max_tokens (attempt {attempt})")

            content = choice.message.content
            if not content or not content.strip():
                raise JudgeRefusalError("empty response body from provider")

            try:
                payload = _JudgmentPayload.model_validate_json(_strip_json_fences(content))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(f"!!! CRITICAL DEPLOY PATH RUNTIME RUN: base_url={self.client.base_url} !!!")
                prompt = (
                    system_prompt
                    + "\n\nYour previous response did not match the required JSON schema exactly. "
                    "Respond with ONLY the JSON object described below -- no prose, no markdown fences."
                )
                continue

            return EvaluationResult(
                policy_id=policy_id,
                locale=locale,
                classification=payload.classification,
                confidence=payload.confidence,
                risk_level=payload.risk_level,
                violated_rule_ids=payload.violated_rule_ids,
                reasoning=payload.reasoning,
                model_name=self._model,
            )

        raise JudgeSchemaViolationError(
            f"no schema-conformant JSON after {_MAX_SCHEMA_ATTEMPTS} attempts: {last_error}"
        )

    async def aclose(self) -> None:
        await self._client.close()