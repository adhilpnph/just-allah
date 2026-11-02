from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from app.rate_limit import limiter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

allowed_origins = os.getenv("ALLOWED_ORIGINS")
if allowed_origins is None or not allowed_origins.strip():
    allowed_origins = "http://localhost:3000"
cors_origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.services.embeddings import load_model

    load_model()
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda _request, _exc: JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait and try again."},
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

from app.routers.ask import router as ask_router
from app.routers.verify import router as verify_router

app.include_router(ask_router)
app.include_router(verify_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello":"World"}
