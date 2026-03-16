from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.embeddings.embedder import embed_text, embed_texts, get_embedding_dimension

# Local chunk storage for keyword search fallback
_local_chunks = []


def get_pinecone_client(api_key: str = None):
    """Create a Pinecone client."""
    key = api_key or PINECONE_API_KEY
    if not key:
        raise ValueError("Pinecone API key is required")
    return Pinecone(api_key=key)


def create_index_if_not_exists(api_key: str = None):
    """Create the Pinecone index if it doesn't exist."""
    pc = get_pinecone_client(api_key)
    dimension = get_embedding_dimension()

    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )
    except Exception:
        pass

    return pc.Index(PINECONE_INDEX_NAME)


def get_index(api_key: str = None):
    """Get the Pinecone index (creates if needed)."""
    return create_index_if_not_exists(api_key)


def upsert_chunks(chunks: List[Dict], api_key: str = None):
    """
    Embed and upload document chunks to Pinecone.
    Also stores chunks locally for keyword search.
    """
    global _local_chunks
    index = get_index(api_key)

    # Store locally for keyword search
    _local_chunks.extend(chunks)

    # Extract texts for batch embedding
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # Prepare vectors for Pinecone
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": chunk["chunk_id"],
            "values": embeddings[i],
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "type": chunk["type"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            },
        })

    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    return len(vectors)


def keyword_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Keyword search through local chunks.
    Finds chunks that contain the most query words.
    """
    if not _local_chunks:
        return []

    import re
    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    query_words = set(query_clean.split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what",
                  "how", "why", "when", "where", "who", "which", "do",
                  "does", "did", "can", "could", "should", "would", "will",
                  "in", "on", "at", "to", "for", "of", "with", "from",
                  "this", "that", "it", "and", "or", "but", "not", "be"}
    content_words = query_words - stop_words

    if not content_words:
        return []

    scored_chunks = []
    for chunk in _local_chunks:
        chunk_text_lower = chunk["text"].lower()
        score = sum(1 for word in content_words if word in chunk_text_lower)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_ids = set()
    for score, chunk in scored_chunks:
        chunk_id = chunk.get("chunk_id", chunk["text"][:50])
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "type": chunk["type"],
                "source": chunk["source"],
                "score": float(score) / len(content_words),
            })
        if len(results) >= top_k:
            break

    return results


def search(query: str, top_k: int = 10, api_key: str = None) -> List[Dict]:
    """
    Hybrid search: Pinecone vector search + keyword search.
    Always returns a list, never None.
    """
    # Vector search
    vector_results = []
    try:
        index = get_index(api_key)
        query_vector = embed_text(query)

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        if results is not None and hasattr(results, 'matches') and results.matches is not None:
            for match in results.matches:
                if match and hasattr(match, 'metadata') and match.metadata:
                    vector_results.append({
                        "text": match.metadata.get("text", ""),
                        "page": match.metadata.get("page", 0),
                        "type": match.metadata.get("type", "text"),
                        "source": match.metadata.get("source", "unknown"),
                        "score": float(match.score) if match.score else 0.0,
                    })
    except Exception as e:
        print(f"Vector search error: {e}")

    # Keyword search
    kw_results = keyword_search(query, top_k=5)
    if kw_results is None:
        kw_results = []

    # Merge — deduplicate by text
    seen_texts = set()
    merged = []

    for r in kw_results:
        text_key = r.get("text", "")[:100]
        if text_key and text_key not in seen_texts:
            seen_texts.add(text_key)
            merged.append(r)

    for r in vector_results:
        text_key = r.get("text", "")[:100]
        if text_key and text_key not in seen_texts:
            seen_texts.add(text_key)
            merged.append(r)

    merged.sort(key=lambda x: x.get("score", 0), reverse=True)

    return merged[:top_k]
def delete_all(api_key: str = None):
    """Delete all vectors from the index. Used for resetting."""
    global _local_chunks
    _local_chunks = []
    try:
        index = get_index(api_key)
        index.delete(delete_all=True)
    except Exception:
        pass