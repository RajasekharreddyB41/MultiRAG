from sentence_transformers import SentenceTransformer
from typing import List
from src.config import EMBEDDING_MODEL

# Cache the model so it only loads once
_model = None


def get_model():
    """Load embedding model (cached after first call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> List[float]:
    """Convert a single text string into a vector."""
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert a list of texts into vectors. Used for batch processing."""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding model. Needed for Pinecone index."""
    model = get_model()
    return model.get_sentence_embedding_dimension()