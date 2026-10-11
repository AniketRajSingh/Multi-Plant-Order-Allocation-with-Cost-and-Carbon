"""FastAPI entrypoint. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.database import get_db


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db = get_db()  # creates the DuckDB file and schema on first start
    logging.getLogger("api").info("DuckDB ready at %s, tables: %s", db.path, db.table_counts())
    yield


app = FastAPI(
    title="Multi-Plant Order Allocation with Cost and Carbon",
    version="1.0.0",
    description="A solver builds the allocation; the LLM only explains it. Section 6 API contract.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
