"""src/services/openai_client.py

Async wrapper around Chat Completions structured outputs, scoped to
EvaluationResult. Owns retry policy explicitly via tenacity; the SDK's
built-in retry is disabled (max_retries=0) to avoid two uncoordinated
backoff loops stacking on top of each other.
"""

from __future__ import annotations

import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.models.taxonomy import EvaluationResult

logger = logging.getLogger(__name__)

# Transient, retry-worthy: rate limits, connection drops, timeouts, upstream 5xx.
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

DEFAULT_MODEL = "gpt-4o-2024-08-06"


class JudgeRefusalError(RuntimeError):
    """Model declined to produce a judgment (safety refusal). Not retryable."""


class JudgeTruncationError(RuntimeError):
    """Structured output could not be completed within max_completion_tokens. Not retryable without raising the cap."""


class OpenAIJudgeClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        max_completion_tokens: int = 800,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self._model = model
        self._max_completion_tokens = max_completion_tokens

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _create(self, *, system_prompt: str, user_prompt: str):
        return await self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=EvaluationResult,
            max_completion_tokens=self._max_completion_tokens,
            temperature=0,
        )

    async def evaluate(self, *, system_prompt: str, user_prompt: str) -> EvaluationResult:
        """Runs one judgment call. Raises JudgeRefusalError / JudgeTruncationError on
        non-retryable structural failures; raises the underlying openai.* exception
        on exhausted retries."""
        try:
            completion = await self._create(system_prompt=system_prompt, user_prompt=user_prompt)
        except LengthFinishReasonError as exc:
            raise JudgeTruncationError(str(exc)) from exc
        except ContentFilterFinishReasonError as exc:
            raise JudgeRefusalError(str(exc)) from exc

        message = completion.choices[0].message

        if message.refusal:
            raise JudgeRefusalError(message.refusal)

        if message.parsed is None:
            # Should not occur given the finish_reason checks above; defensive guard
            # against SDK/version drift silently returning an unparsed message.
            raise ValueError("parse succeeded with no exception but message.parsed is None")

        return message.parsed

    async def aclose(self) -> None:
        await self._client.close()