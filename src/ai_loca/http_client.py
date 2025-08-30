from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .errors import HTTPRequestError

logger = logging.getLogger(__name__)


def _retry_decorator(retries: int):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max(1, retries)),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.ReadTimeout, httpx.ConnectTimeout)),
    )


class HttpClient:
    def __init__(self, timeout: float = 60.0, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self._client = httpx.Client(timeout=self.timeout)

    def close(self):
        self._client.close()

    @_retry_decorator(retries=3)
    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        # retry count overridden at instance call site by wrapper
        return self._client.request(method, url, **kwargs)

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Any] = None,
        data: Optional[Any] = None,
        content: Optional[bytes] = None,
        params: Optional[Dict[str, Any]] = None,
        expected_status: int | tuple[int, ...] = (200, 201),
    ) -> httpx.Response:
        try:
            logger.debug("HTTP %s %s", method, url)
            # Use instance-configured retries by wrapping the method
            # We achieve this by temporarily setting the decorator attribute
            # Note: tenacity decorators are created at definition time; for per-instance setting,
            # we can just loop attempts manually here.
            attempts = max(1, self.retries)
            last_exc: Optional[Exception] = None
            for i in range(attempts):
                try:
                    resp = self._client.request(
                        method,
                        url,
                        headers=headers,
                        json=json,
                        data=data,
                        content=content,
                        params=params,
                    )
                    if isinstance(expected_status, int):
                        ok = resp.status_code == expected_status
                    else:
                        ok = resp.status_code in expected_status
                    if not ok:
                        logger.warning("HTTP %s returned %s: %s", url, resp.status_code, resp.text[:500])
                        resp.raise_for_status()
                    return resp
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last_exc = e
                    logger.warning("HTTP attempt %s/%s failed: %s", i + 1, attempts, e)
            if last_exc:
                raise last_exc
            raise HTTPRequestError(f"Request failed without exception: {method} {url}")
        except Exception as e:  # noqa: BLE001
            raise HTTPRequestError(str(e)) from e
