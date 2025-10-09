"""
Minimal OpenAI-compatible shim for local testing.

Endpoints implemented (subset):
 - POST /v1/chat/completions

Behaviour: Echoes the last user message content in a deterministic response
with fixed ids/timestamps. This allows smoke tests of OpenAI-compatible
clients without external network access.

Optional deps: fastapi, uvicorn, pydantic

Run: python -m bsde_dsgE.compat_server
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, TypedDict

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except Exception:  # pragma: no cover - keep import-time errors silent for optional dep
    FastAPI = None  # type: ignore
    JSONResponse = None  # type: ignore
    BaseModel = object  # type: ignore


class ChatMessage(TypedDict, total=False):
    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):  # type: ignore[misc]
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


def _echo_choice(content: str) -> Dict[str, Any]:
    return {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": content,
        },
    }


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "fastapi is not installed. Install optional deps to run the compat server."
        )
    app = FastAPI(title="OpenAI-compatible Shim", version="0.1.0")

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {"data": [{"id": "echo-mini", "object": "model"}], "object": "list"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionsRequest) -> Any:  # type: ignore[valid-type]
        # Find last user content
        last_user = ""
        for m in reversed(req.messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                last_user = c if isinstance(c, str) else str(c)
                break
        now = int(time.time())
        resp = {
            "id": f"chatcmpl-echo-{now}",
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [_echo_choice(last_user)],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(last_user.split()), "total_tokens": len(last_user.split())},
        }
        return JSONResponse(resp)  # type: ignore[call-arg]

    return app


def main() -> None:
    if FastAPI is None:
        raise SystemExit(
            "fastapi/uvicorn not installed. Try: pip install 'fastapi[standard]' uvicorn"
        )
    import uvicorn  # type: ignore

    host = os.environ.get("COMPAT_HOST", "127.0.0.1")
    port = int(os.environ.get("COMPAT_PORT", "8001"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
