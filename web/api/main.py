"""FastAPI application entry point for the PPG-Agent web layer.

Run from the project root:
    uvicorn web.api.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .deps import enable_wal_once
from .routers import analysis, chat, data


@asynccontextmanager
async def lifespan(app: FastAPI):
    enable_wal_once()
    yield


app = FastAPI(
    title="PPG-Agent API",
    description="Multi-agent PPG cardiovascular phenotyping — web layer",
    version="0.1.0",
    lifespan=lifespan,
)

# Vite dev server origin (demo deployment serves the built frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix=config.API_PREFIX)
app.include_router(analysis.router, prefix=config.API_PREFIX)
app.include_router(chat.router, prefix=config.API_PREFIX)


@app.get("/api/health")
def health():
    return {"status": "ok"}
