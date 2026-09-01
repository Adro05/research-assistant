"""
FastAPI application entrypoint.

This is a project-foundation scaffold. No feature routers (projects,
documents, search, RAG, etc.) are wired in yet — those are added in
their own milestones.
"""

from fastapi import FastAPI

from backend.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered research assistant — API backend.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic liveness check used to verify the backend is running."""
    return {"status": "ok", "app_name": settings.app_name}