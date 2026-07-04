from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import os
import logging
import random
import json

import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_MAX_CONCURRENCY = int(os.getenv("TEACHER_CONCURRENCY", "128"))

_TEACHER_MODEL_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENCY)

_TEACHER_MODEL_NAME = os.getenv("TEACHER_MODEL_NAME")
assert _TEACHER_MODEL_NAME is not None

_TEACHER_MODEL_CLIENT = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)


def _parse_optional_bool_env(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None or value.strip() in ("", "null", "None"):
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _build_teacher_extra_body() -> Optional[Dict[str, Any]]:
    """Build provider-specific extra OpenAI request body for teacher calls."""
    raw_kwargs = os.getenv("TEACHER_CHAT_TEMPLATE_KWARGS")
    if raw_kwargs is not None and raw_kwargs.strip() not in ("", "null", "None"):
        chat_template_kwargs = json.loads(raw_kwargs)
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("TEACHER_CHAT_TEMPLATE_KWARGS must decode to a JSON object")
        return {"chat_template_kwargs": chat_template_kwargs}

    enable_thinking = _parse_optional_bool_env("TEACHER_ENABLE_THINKING")
    if enable_thinking is None:
        return None
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


_TEACHER_EXTRA_BODY = _build_teacher_extra_body()


async def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
    # first attempt = 1 -> 0.5s to 0.7s
    # second attempt = 2 -> 1.0s to 1.2s
    # third attempt = 3 -> 2.0s to 2.2s
    
    delay = min(cap, base * (2 ** (attempt - 1)))
    delay += random.uniform(0, 0.2)
    await asyncio.sleep(delay)

async def call_teacher_with_retry(
    message: str,
    parse_fn: Callable[[str], Any],
    *,
    temperature_schedule: Sequence[float] = (0.0, 0.3, 0.7),
    max_attempts_per_temperature: int = 1,
    timeout: int = 1800,
    log_prefix: str = "LLM_JUDGE",
) -> Tuple[Optional[Any], str]:
    last_reply: str = ""
    global_attempt = 0

    for temperature in temperature_schedule:
        for _ in range(1, max_attempts_per_temperature + 1):
            global_attempt += 1
            try:
                async with _TEACHER_MODEL_SEMAPHORE:
                    request_kwargs = {
                        "model": _TEACHER_MODEL_NAME,
                        "messages": [{"role": "user", "content": message}],
                        "temperature": temperature,
                        "timeout": timeout,
                    }
                    if _TEACHER_EXTRA_BODY is not None:
                        request_kwargs["extra_body"] = _TEACHER_EXTRA_BODY
                    response = await _TEACHER_MODEL_CLIENT.chat.completions.create(
                        **request_kwargs
                    )
                reply = (response.choices[0].message.content or "").strip()
                last_reply = reply
            except Exception as e:
                logger.warning(
                    f"{log_prefix} request error "
                    f"(temperature={temperature}, attempt={global_attempt}): "
                    f"{type(e).__name__}: {e}"
                )
                await _backoff_sleep(global_attempt)
                continue

            try:
                parsed = parse_fn(reply)
                return parsed, reply
            except Exception as e:
                logger.warning(
                    f"{log_prefix} parse error\n"
                    f"(temperature={temperature}, attempt={global_attempt})\n"
                    f"Error: {e}\n"
                    f"Reply: {reply}"
                )
                await _backoff_sleep(global_attempt)
                continue

    return None, last_reply
