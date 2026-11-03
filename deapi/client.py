"""Minimal client for the deAPI v2 async API (submit -> poll -> download)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import requests

from . import __version__

log = logging.getLogger("deAPI")

BASE_URL = "https://api.deapi.ai/api/v2"
REGISTER_URL = "https://app.deapi.ai/register"
KEYS_URL = "https://app.deapi.ai/settings/api-keys"
USER_AGENT = f"comfyui-deapi/{__version__}"


class DeapiError(RuntimeError):
    """Error raised for any deAPI failure. The message is shown in the ComfyUI UI."""


def resolve_api_key(api_key: Optional[str]) -> str:
    key = (api_key or "").strip() or os.environ.get("DEAPI_API_KEY", "").strip()
    if not key:
        raise DeapiError(
            "No deAPI API key. Paste it into the node's 'api_key' input or set the "
            f"DEAPI_API_KEY environment variable before starting ComfyUI. "
            f"Get a key at {REGISTER_URL}"
        )
    return key


def _describe_http_error(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        body = None
    message = ""
    code = ""
    if isinstance(body, dict):
        message = str(body.get("message") or body.get("error") or "")
        code = str(body.get("code") or "")
        errors = body.get("errors")
        if isinstance(errors, dict) and errors:
            texts = []
            for msgs in errors.values():
                texts.extend(map(str, msgs) if isinstance(msgs, list) else [str(msgs)])
            # Laravel puts the first error in "message"; list only the ones not already there.
            extra = [t for t in texts if t not in message]
            if extra:
                message = f"{message} {' '.join(extra)}".strip()
    if not message:
        message = (resp.text or "").strip()[:500] or resp.reason or "no details"
    message = message.rstrip(".")

    status = resp.status_code
    low = message.lower()
    if status == 401:
        hint = f" Check your API key ({KEYS_URL})."
    elif "balance" in low or "insufficient" in low or "credit" in low or status == 402:
        hint = " Your deAPI balance is too low - top up at https://app.deapi.ai."
    elif status == 403:
        hint = " This key may not call this endpoint (role or IP whitelist)."
    elif status == 429:
        hint = " Rate limit hit - wait a moment and try again."
    else:
        hint = ""
    code_part = f" [{code}]" if code else ""
    return f"deAPI HTTP {status}{code_part}: {message}.{hint}"


class DeapiClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL):
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            }
        )

    # ------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, *, retries: int = 3, **kwargs) -> dict:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        kwargs.setdefault("timeout", 120)
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                # Only GETs are retried on network errors: a resubmitted POST could be charged twice.
                if method == "GET" and attempt <= retries:
                    time.sleep(2 * attempt)
                    continue
                raise DeapiError(f"deAPI network error calling {path}: {exc}") from exc

            if resp.status_code == 429 and attempt <= retries:
                wait = resp.headers.get("Retry-After")
                try:
                    delay = min(float(wait), 60.0) if wait else 5.0 * attempt
                except ValueError:
                    delay = 5.0 * attempt
                log.warning("deAPI rate limit hit, retrying in %.0fs", delay)
                time.sleep(delay)
                continue
            if resp.status_code >= 500 and method == "GET" and attempt <= retries:
                time.sleep(2 * attempt)
                continue
            if resp.status_code >= 400:
                raise DeapiError(_describe_http_error(resp))
            try:
                return resp.json()
            except ValueError as exc:
                raise DeapiError(f"deAPI returned a non-JSON response from {path}: {resp.text[:300]}") from exc

    def submit(self, path: str, *, json: Optional[dict] = None, data: Optional[dict] = None,
               files: Any = None) -> str:
        if data is not None:
            data = {k: _form_value(v) for k, v in data.items() if v is not None}
        body = self._request("POST", path, json=json, data=data, files=files, timeout=300)
        try:
            return str(body["data"]["request_id"])
        except (KeyError, TypeError) as exc:
            raise DeapiError(f"Unexpected deAPI response from {path}: {str(body)[:300]}") from exc

    def job(self, request_id: str) -> dict:
        body = self._request("GET", f"/jobs/{request_id}", timeout=60)
        return body.get("data") or {}

    def wait(self, request_id: str, *, timeout: float = 600, label: str = "job",
             on_progress: Optional[Callable[[float], None]] = None,
             check_interrupt: Optional[Callable[[], None]] = None) -> dict:
        """Poll a job until done/error. Raises DeapiError on failure or timeout."""
        start = time.monotonic()
        delay = 1.5
        last_status = None
        last_log = 0.0
        while True:
            if check_interrupt:
                check_interrupt()
            job = self.job(request_id)
            status = job.get("status")
            progress = job.get("progress")
            if on_progress and isinstance(progress, (int, float)):
                on_progress(float(progress))
            elapsed = time.monotonic() - start
            if status != last_status or elapsed - last_log > 30:
                pct = f" {progress:.0f}%" if isinstance(progress, (int, float)) and progress else ""
                log.info("deAPI %s %s: %s%s (%.0fs)", label, request_id, status, pct, elapsed)
                last_status, last_log = status, elapsed
            if status == "done":
                price = job.get("price")
                if isinstance(price, dict):
                    price = price.get("amount")
                log.info("deAPI %s %s done in %.1fs%s", label, request_id, elapsed,
                         f", cost {price:.5f} credits" if isinstance(price, (int, float)) else "")
                return job
            if status == "error":
                code = job.get("error_code") or "ERROR"
                reason = job.get("error_reason") or "no reason given"
                extra = []
                if job.get("retryable"):
                    extra.append("retrying may succeed")
                if job.get("refunded"):
                    extra.append("the charge was refunded")
                tail = f" ({', '.join(extra)})" if extra else ""
                raise DeapiError(f"deAPI {label} failed [{code}]: {reason}{tail}. Request id: {request_id}")
            if elapsed > timeout:
                raise DeapiError(
                    f"deAPI {label} timed out after {timeout:.0f}s (last status: {status}). "
                    f"The job may still finish - request id: {request_id}"
                )
            time.sleep(delay)
            delay = min(delay * 1.3, 10.0)

    def download(self, url: str, timeout: float = 300) -> bytes:
        try:
            # Result URLs are pre-signed; do not send the API key to the storage host.
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            raise DeapiError(f"Could not download deAPI result: {exc}") from exc
        if resp.status_code >= 400:
            raise DeapiError(f"Could not download deAPI result: HTTP {resp.status_code}")
        return resp.content

    def run(self, path: str, *, label: str, timeout: float = 600, json: Optional[dict] = None,
            data: Optional[dict] = None, files: Any = None,
            on_progress: Optional[Callable[[float], None]] = None,
            check_interrupt: Optional[Callable[[], None]] = None) -> dict:
        rid = self.submit(path, json=json, data=data, files=files)
        log.info("deAPI %s submitted: %s", label, rid)
        return self.wait(rid, timeout=timeout, label=label, on_progress=on_progress,
                         check_interrupt=check_interrupt)


def _form_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
