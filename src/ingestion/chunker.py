from typing import List, Dict
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks.
    Uses smaller chunks for better retrieval accuracy.
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    if len(text) <= size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "? ", "! ", " "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def chunk_documents(pages: List[Dict]) -> List[Dict]:
    """
    Take extracted pages and split them into chunks.
    Each chunk keeps metadata about its source page.
    Also creates keyword-rich mini chunks for better search.
    """
    all_chunks = []

    for page in pages:
        text = page.get("text", "")
        if not text:
            continue

        # Regular chunks
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page": page["page"],
                "type": page.get("type", "text"),
                "source": page.get("source", "unknown"),
                "chunk_index": i,
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}",
            })

        # Create a page summary chunk for better retrieval
        # This helps find answers when the query doesn't match exact words
        if len(text) > 200:
            summary = text[:300].strip()
            all_chunks.append({
                "text": f"Page {page['page']} summary: {summary}",
                "page": page["page"],
                "type": "summary",
                "source": page.get("source", "unknown"),
                "chunk_index": 999,
                "chunk_id": f"{page['source']}_p{page['page']}_summary",
            })

    return all_chunks