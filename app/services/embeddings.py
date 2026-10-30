from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    embeddings = _get_model().encode(
        [f"passage: {text}" for text in texts],
        batch_size=64,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def load_model() -> None:
    _get_model()


def embed_passage(text: str) -> list[float]:
    return _get_model().encode(
        f"passage: {text}",
        normalize_embeddings=True,
    ).tolist()


def embed_query(text: str) -> list[float]:
    return _get_model().encode(
        f"query: {text}",
        normalize_embeddings=True,
    ).tolist()
