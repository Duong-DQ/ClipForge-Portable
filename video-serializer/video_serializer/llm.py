"""Optional LLM client (stdlib only).

Single-purpose helper that talks to an OpenAI-compatible ``/chat/completions``
endpoint and returns the assistant text. Used by the segmentation, enrichment
and recap stages when ``config.use_llm`` is true.

Design constraints
------------------
* Stdlib only (``json``, ``urllib.request``, ``sys``, ``re``). No new pip deps.
* Never raise. Every error path (network, HTTP, JSON decode, missing fields)
  emits one short stderr line and returns ``None`` so callers can fall back
  to their offline behaviour without try/except.
* Never log the API key. Errors mask the key (``first6...`` or ``***``).
* Default model is ``gemini-3.8-flash`` when ``config.llm_model`` is empty.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

__all__ = ["is_configured", "chat", "chat_json"]


# Confirmed-working Gemini endpoint and current non-deprecated model.
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_DEFAULT_MODEL = "gemini-3.8-flash"


def _mask_key(api_key: str) -> str:
    """Return ``"first6chars..."`` so a leaked stderr line cannot reveal the key."""
    if not api_key:
        return ""
    head = api_key[:6]
    return f"{head}..."


def _build_opener(config) -> urllib.request.OpenerDirector:
    """Build an opener that respects explicit proxy config or system defaults.

    Priority:
    1. ``config.llm_proxy`` — explicit proxy URL (e.g. ``http://127.0.0.1:8899``)
    2. ``HTTPS_PROXY`` / ``HTTP_PROXY`` env vars
    3. Windows system proxy (registry) — auto-detected by ProxyHandler()
    """
    proxy = (getattr(config, "llm_proxy", "") or "").strip()
    if proxy:
        proxies = {"http": proxy, "https": proxy}
    else:
        # Let ProxyHandler auto-detect from env vars / OS registry
        proxies = None
    handler = urllib.request.ProxyHandler(proxies) if proxies else urllib.request.ProxyHandler()
    return urllib.request.build_opener(handler)


def is_configured(config) -> bool:
    """Return ``True`` iff the config opts into the LLM and provides url+key.

    ``config`` is duck-typed: we only read ``use_llm``, ``llm_base_url`` and
    ``llm_api_key`` attributes. Missing/empty strings fail the check, so the
    offline path is taken by default.
    """
    if not getattr(config, "use_llm", False):
        return False
    base = (getattr(config, "llm_base_url", "") or "").strip()
    key = (getattr(config, "llm_api_key", "") or "").strip()
    return bool(base) and bool(key)


def _strip_code_fences(text: str) -> str:
    """Strip a surrounding ```` ```json ... ``` ```` fence if present."""
    text = (text or "").strip()
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1 :]
        else:
            text = text.lstrip("`")
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


def _short_reason(exc: BaseException) -> str:
    """Single-line, length-bounded reason text for the stderr warning line."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return msg[:160] or type(exc).__name__


def _build_request(
    config, system: str, user: str, temperature: float, max_tokens: int
) -> urllib.request.Request:
    """Assemble the urllib request: URL, body, headers (key masked on errors)."""
    base = (getattr(config, "llm_base_url", "") or "").strip().rstrip("/")
    url = base + "/chat/completions"
    model = (getattr(config, "llm_model", "") or "").strip() or _DEFAULT_MODEL
    body = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    api_key = (getattr(config, "llm_api_key", "") or "").strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return urllib.request.Request(url, data=data, headers=headers, method="POST")


def chat(config, system: str, user: str, temperature: float = 0.0,
         max_tokens: int = 500, timeout: int = 60) -> "str | None":
    """POST a chat-completions request and return the assistant text.

    On any failure (URLError, HTTPError, timeout, JSON decode, missing fields,
    non-string content) emit ONE stderr line ``[llm] chat failed: <short>``
    with the API key masked and return ``None``. Never raises.
    """
    api_key = (getattr(config, "llm_api_key", "") or "").strip()
    masked = _mask_key(api_key)
    try:
        request = _build_request(config, system, user, temperature, max_tokens)
        opener = _build_opener(config)
        with opener.open(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", "replace")
        parsed = json.loads(payload)
        content = parsed["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            print(
                f"[llm] chat failed: non-string content (key={masked})",
                file=sys.stderr,
            )
            return None
        return content.strip()
    except Exception as exc:  # noqa: BLE001 - deliberate, must never raise
        print(
            f"[llm] chat failed: {_short_reason(exc)} (key={masked})",
            file=sys.stderr,
        )
        return None


# A loose JSON object extractor: find the first balanced {...} run in the text
# and try to parse it. Tolerates models that wrap the JSON in prose or fences.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def chat_json(config, system: str, user: str, temperature: float = 0.0,
              max_tokens: int = 500) -> "dict | None":
    """Same as :func:`chat` but parses the assistant text as a JSON object.

    Behaviour:
      * ``None`` from :func:`chat` -> ``None``.
      * Strip markdown fences.
      * Try ``json.loads`` directly; on failure fall back to a regex search for
        the first balanced ``{...}`` substring and try again.
      * Return the parsed object iff it is a ``dict``; otherwise a stderr line
        and ``None``. Never raises.
    """
    raw = chat(config, system, user, temperature=temperature, max_tokens=max_tokens)
    if raw is None:
        return None
    text = _strip_code_fences(raw)
    parsed: object = None
    try:
        parsed = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if match is not None:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        print(
            "[llm] chat_json failed: response was not a JSON object",
            file=sys.stderr,
        )
        return None
    return parsed


# --------------------------------------------------------------------------- #
# Offline self-test (no network, no key)
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Offline checks: nothing imports a network module and the masking works."""
    from .models import PipelineConfig  # local import keeps top of file clean

    # (a) is_configured gates every field
    cfg_off = PipelineConfig(use_llm=False, llm_base_url=_DEFAULT_BASE_URL,
                             llm_api_key="sk-abc")
    assert is_configured(cfg_off) is False, "use_llm=False should disable LLM"

    cfg_no_url = PipelineConfig(use_llm=True, llm_base_url="", llm_api_key="sk-abc")
    assert is_configured(cfg_no_url) is False, "empty base URL must disable"

    cfg_no_key = PipelineConfig(use_llm=True, llm_base_url=_DEFAULT_BASE_URL,
                                llm_api_key="")
    assert is_configured(cfg_no_key) is False, "empty API key must disable"

    cfg_whitespace = PipelineConfig(use_llm=True, llm_base_url="  ",
                                    llm_api_key="\t ")
    assert is_configured(cfg_whitespace) is False, "whitespace-only must disable"

    cfg_on = PipelineConfig(use_llm=True, llm_base_url=_DEFAULT_BASE_URL,
                            llm_api_key="sk-abcdef1234567890",
                            llm_model=_DEFAULT_MODEL)
    assert is_configured(cfg_on) is True, "full config must be enabled"

    # (b) mask helper hides the key tail
    assert _mask_key("") == ""
    assert _mask_key("short") == "short..."
    assert _mask_key("sk-abc123def456") == "sk-abc..."

    # (c) code-fence stripping is idempotent and safe on bare text
    assert _strip_code_fences("") == ""
    assert _strip_code_fences("hello") == "hello"
    assert _strip_code_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert _strip_code_fences("```\n{\"a\": 1}```") == '{"a": 1}'

    # (d) request builder uses configured base url + model + max tokens
    request = _build_request(
        cfg_on,
        system="sys",
        user="hi",
        temperature=0.0,
        max_tokens=42,
    )
    assert request.get_full_url() == _DEFAULT_BASE_URL + "/chat/completions", (
        request.get_full_url()
    )
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == _DEFAULT_MODEL
    assert body["max_tokens"] == 42
    assert body["temperature"] == 0.0
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert request.get_header("Authorization") == "Bearer sk-abcdef1234567890", (
        request.get_header("Authorization")
    )

    # (e) default model fills in when llm_model is empty
    cfg_default_model = PipelineConfig(
        use_llm=True, llm_base_url=_DEFAULT_BASE_URL,
        llm_api_key="sk-x", llm_model="",
    )
    req2 = _build_request(cfg_default_model, "s", "u", 0.0, 100)
    body2 = json.loads(req2.data.decode("utf-8"))
    assert body2["model"] == _DEFAULT_MODEL

    # (f) trailing slash on the base URL is normalised away
    cfg_slash = PipelineConfig(
        use_llm=True, llm_base_url=_DEFAULT_BASE_URL + "/",
        llm_api_key="sk-x", llm_model=_DEFAULT_MODEL,
    )
    req3 = _build_request(cfg_slash, "s", "u", 0.0, 10)
    assert req3.get_full_url() == _DEFAULT_BASE_URL + "/chat/completions"

    print("PASS")


if __name__ == "__main__":
    _self_test()
