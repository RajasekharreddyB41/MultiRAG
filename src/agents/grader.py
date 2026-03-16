from src.llm.groq_client import get_groq_client
from src.config import GROQ_MODEL
from typing import List, Dict


def grade_retrieval(query: str, chunks: List[Dict], api_key: str = None) -> Dict:
    """
    Grade whether retrieved chunks are relevant to the query.
    Returns a dict with:
      - is_relevant: True if chunks can answer the query
      - score: 0-10 relevance score
      - filtered_chunks: only the relevant chunks
    """
    if not chunks:
        return {
            "is_relevant": False,
            "score": 0,
            "filtered_chunks": [],
            "feedback": "No documents found.",
        }

    # Combine chunk texts for grading
    context = "\n\n".join([
        f"[Page {c['page']}] {c['text'][:300]}"
        for c in chunks[:5]
    ])

    try:
        client = get_groq_client(api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a retrieval quality grader.
Given a question and retrieved context, evaluate if the context contains enough information to answer the question.

Respond in EXACTLY this format:
SCORE: [0-10]
RELEVANT: [YES/NO]
FEEDBACK: [one sentence explaining why]

Score guide:
- 0-3: Context is completely irrelevant
- 4-6: Context is partially relevant but may not fully answer
- 7-10: Context is highly relevant and can answer the question""",
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nRetrieved Context:\n{context}",
                },
            ],
            temperature=0,
            max_tokens=100,
        )

        result = response.choices[0].message.content.strip()

        # Parse the response
        score = _extract_score(result)
        is_relevant = score >= 5

        # Filter chunks by individual relevance
        filtered = _filter_relevant_chunks(query, chunks, api_key)

        return {
            "is_relevant": is_relevant,
            "score": score,
            "filtered_chunks": filtered if filtered else chunks[:3],
            "feedback": result,
        }

    except Exception as e:
        # If grading fails, pass all chunks through
        return {
            "is_relevant": True,
            "score": 5,
            "filtered_chunks": chunks,
            "feedback": f"Grading skipped: {str(e)}",
        }


def _extract_score(result: str) -> int:
    """Extract the numeric score from grader response."""
    for line in result.split("\n"):
        if "SCORE" in line.upper():
            # Find the number in the line
            for word in line.split():
                try:
                    score = int(word.strip(":").strip())
                    if 0 <= score <= 10:
                        return score
                except ValueError:
                    continue
    return 5  # Default middle score


def _filter_relevant_chunks(query: str, chunks: List[Dict], api_key: str = None) -> List[Dict]:
    """
    Simple relevance filter using keyword overlap.
    Keeps chunks that share words with the query.
    """
    query_words = set(query.lower().split())
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what",
                  "how", "why", "when", "where", "who", "which", "do",
                  "does", "did", "can", "could", "should", "would", "will",
                  "in", "on", "at", "to", "for", "of", "with", "from", "this", "that"}
    query_words = query_words - stop_words

    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(query_words & chunk_words)
        scored_chunks.append((overlap, chunk))

    # Sort by overlap score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return chunks with at least some overlap
    filtered = [chunk for score, chunk in scored_chunks if score > 0]

    return filtered if filtered else chunks[:3]


def should_retry(grade_result: Dict) -> bool:
    """Decide if we should re-retrieve with a refined query."""
    return grade_result["score"] < 4


def refine_query(original_query: str, feedback: str, api_key: str = None) -> str:
    """
    Create a better search query based on grader feedback.
    Used when first retrieval is low quality.
    """
    try:
        client = get_groq_client(api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a search query optimizer.
The user's original question didn't find good results.
Rewrite it as a better search query - use different keywords,
be more specific, or try alternative phrasing.
Respond with ONLY the improved query, nothing else.""",
                },
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\nFeedback: {feedback}",
                },
            ],
            temperature=0.3,
            max_tokens=50,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return original_query