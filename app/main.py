from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.services.embeddings import load_model

    load_model()
    yield


app = FastAPI(lifespan=lifespan)

from app.routers.ask import router as ask_router
from app.routers.verify import router as verify_router

app.include_router(ask_router)
app.include_router(verify_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello":"World"}