from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def build_retry(retries: int, exceptions: tuple[type[BaseException], ...]):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max(1, retries)),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        retry=retry_if_exception_type(exceptions),
    )

