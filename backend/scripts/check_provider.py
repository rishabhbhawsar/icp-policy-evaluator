"""scripts/check_provider.py

Preflight check: confirms the configured base_url/api_key/model can actually
reach the provider and that the configured model exists on that account,
before running the full test harness. Run directly:

    python -m scripts.check_provider
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI, AuthenticationError, OpenAIError

from src.core.config import get_settings


async def main() -> None:
    settings = get_settings()

    print(f"Configured base URL : {settings.openai_base_url or '(default: api.openai.com)'}")
    print(f"Configured model     : {settings.openai_model}")

    client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
    )

    try:
        response = await client.models.list()
    except AuthenticationError as exc:
        print(f"\n[PREFLIGHT FAILED] Authentication rejected: {exc}")
        return
    except OpenAIError as exc:
        print(f"\n[PREFLIGHT FAILED] Could not reach provider: {exc}")
        return
    except Exception as exc:  # non-JSON response (e.g. an HTML page) breaks the SDK's own parser
        print(f"\n[PREFLIGHT FAILED] Unexpected response while parsing ({type(exc).__name__}: {exc})")
        print(
            "This usually means base_url pointed at a page that isn't the API endpoint "
            "(e.g. a provider's website instead of its /api/v1 or /openai/v1 path) -- "
            "double-check OPENAI_BASE_URL against the provider's docs."
        )
        return
    finally:
        await client.close()

    model_ids = sorted(m.id for m in response.data)
    print(f"\n[PREFLIGHT SUCCESS] {len(model_ids)} models visible to this account.")
    print("First 10:", model_ids[:10])

    if settings.openai_model in model_ids:
        print(f"\nConfigured model '{settings.openai_model}' IS in that list.")
    else:
        print(
            f"\n[WARNING] Configured model '{settings.openai_model}' is NOT in the list above. "
            "Update OPENAI_MODEL to one of the values printed here before running the test harness."
        )


if __name__ == "__main__":
    asyncio.run(main())