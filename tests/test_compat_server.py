import socket
import time

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _has_fastapi():
    try:
        import fastapi  # noqa: F401

        import uvicorn  # noqa: F401
        import httpx  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_fastapi(), reason="fastapi/uvicorn/httpx not installed")


def test_chat_completions_echo():
    from bsde_dsgE.compat_server import create_app
    import threading
    import uvicorn
    import httpx

    port = _free_port()
    app = create_app()

    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(0.5)

    with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo-mini",
                "messages": [
                    {"role": "system", "content": "You echo."},
                    {"role": "user", "content": "Hello world"},
                ],
            },
            timeout=5.0,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello world"
