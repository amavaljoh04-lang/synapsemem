"""Thin async client for the Ollama HTTP API.

We intentionally depend on no SDK — the Ollama HTTP surface is tiny and
stable. Using httpx keeps us async-friendly inside FastAPI without adding
dead weight to the image.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class OllamaError(RuntimeError):
    """Raised when Ollama returns a non-2xx response or a malformed payload."""


class OllamaClient:
    """Minimal Ollama client: chat completion + embeddings + tags."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    async def tags(self) -> list[dict]:
        """Return the list of models currently installed on the Ollama server."""
        async with await self._client() as http:
            r = await http.get("/api/tags")
            if r.status_code != 200:
                raise OllamaError(f"/api/tags → {r.status_code}: {r.text[:200]}")
            return r.json().get("models", [])

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict | None = None,
    ) -> dict:
        """Single-shot (non-streaming) chat completion.

        Returns the full Ollama response dict. The answer text lives at
        ``response["message"]["content"]``.
        """
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        async with await self._client() as http:
            r = await http.post("/api/chat", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"/api/chat → {r.status_code}: {r.text[:500]}")
            data = r.json()
            if "message" not in data:
                raise OllamaError(f"missing 'message' in chat response: {data}")
            return data

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat tokens as they arrive. Yields text chunks."""
        import json

        payload: dict = {"model": model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        async with await self._client() as http, http.stream(
            "POST", "/api/chat", json=payload
        ) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise OllamaError(
                    f"/api/chat stream → {r.status_code}: {body[:500]!r}"
                )
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    yield piece
                if chunk.get("done"):
                    return

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        suffix: str | None = None,
        options: dict | None = None,
        raw: bool = False,
    ) -> dict:
        """Raw completion via ``/api/generate``.

        Used by the CrossCodeEval harness to hit fill-in-the-middle
        models (``qwen2.5-coder``, ``deepseek-coder``, ``starcoder2``)
        with their native FIM tokens rather than a chat wrapper. The
        response text lives at ``response["response"]``.
        """
        payload: dict = {"model": model, "prompt": prompt, "stream": False}
        if suffix is not None:
            payload["suffix"] = suffix
        if raw:
            payload["raw"] = True
        if options:
            payload["options"] = options
        async with await self._client() as http:
            r = await http.post("/api/generate", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"/api/generate → {r.status_code}: {r.text[:500]}")
            data = r.json()
            if "response" not in data:
                raise OllamaError(f"missing 'response' in generate payload: {data}")
            return data

    async def embed(self, model: str, text: str) -> list[float]:
        """Return an embedding vector for ``text`` using ``model``."""
        async with await self._client() as http:
            r = await http.post(
                "/api/embeddings", json={"model": model, "prompt": text}
            )
            if r.status_code != 200:
                raise OllamaError(f"/api/embeddings → {r.status_code}: {r.text[:200]}")
            data = r.json()
            vec = data.get("embedding") or []
            if not isinstance(vec, list):
                raise OllamaError(f"malformed embedding payload: {data}")
            return [float(x) for x in vec]
