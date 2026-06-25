from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from prism_serve import __version__
from prism_serve.config import settings

app = FastAPI(
    title="prism-serve",
    version=__version__,
    summary="Kubernetes control plane for disaggregated LLM serving",
)


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe — process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness probe.

    Once the router/scheduler land this will reflect whether at least one healthy
    worker is reachable. For now the gateway is always ready.
    """
    return {"status": "ready"}


@app.post("/v1/chat/completions")
def chat_completions() -> JSONResponse:
    """OpenAI-compatible inference entry point (not yet implemented)."""
    return JSONResponse(
        status_code=501,
        content={"error": "not_implemented", "detail": "routing lands in a later milestone"},
    )


def main() -> None:
    """Console-script entry point: run the gateway with uvicorn."""
    import uvicorn

    uvicorn.run(
        "prism_serve.gateway.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
