"""
Ollama Client Adapter
======================
Minimal wrapper to call a local Ollama server using the same interface
as ``google.genai.Client.models.generate_content``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Lightweight response wrapper with a ``text`` attribute."""

    text: str
    raw: dict[str, Any]


class OllamaClient:
    """HTTP client for the local Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_content(self, model: str, contents: str) -> OllamaResponse:
        """Call Ollama's /api/generate endpoint and return a response object."""
        payload = {
            "model": model,
            "prompt": contents,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            msg = f"{exc.code} {exc.reason}"
            logger.warning("Ollama HTTP error: %s", msg)
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"ollama_connection_error: {exc.reason}"
            logger.warning("Ollama connection error: %s", msg)
            raise RuntimeError(msg) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("Ollama invalid JSON response: %s", body[:200])
            raise RuntimeError("ollama_invalid_json_response") from exc

        text = parsed.get("response", "")
        return OllamaResponse(text=text, raw=parsed)
