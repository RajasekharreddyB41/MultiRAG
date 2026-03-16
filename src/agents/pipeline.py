from typing import Dict, List, Generator
from src.agents.router import route_query, get_route_description


MAX_RETRIES = 2


def run_pipeline(
    query: str,
    chat_history: list = None,
    has_documents: bool = True,
    images: dict = None,
    api_keys: dict = None,
) -> Dict:
    """Main RAG pipeline."""
    keys = api_keys or {}
    groq_key = keys.get("groq", "")
    google_key = keys.get("google", "")
    pinecone_key = keys.get("pinecone", "")
    namespace = keys.get("namespace", "default")

    route = route_query(query, has_documents, groq_key)

    if route == "general":
        return _handle_general(query, chat_history, groq_key)

    if route == "image" and images:
        return _handle_image(query, images, google_key)

    return _handle_rag(query, chat_history, groq_key, pinecone_key, namespace)


def run_pipeline_stream(
    query: str,
    chat_history: list = None,
    has_documents: bool = True,
    images: dict = None,
    api_keys: dict = None,
) -> Generator:
    """Streaming version of the pipeline."""
    keys = api_keys or {}
    groq_key = keys.get("groq", "")
    google_key = keys.get("google", "")
    pinecone_key = keys.get("pinecone", "")
    namespace = keys.get("namespace", "default")

    # Step 1: Route
    route = route_query(query, has_documents, groq_key)
    yield {"type": "status", "message": get_route_description(route)}

    # Step 2: General questions — no document search needed
    if route == "general":
        yield {"type": "status", "message": "Generating answer..."}
        try:
            from src.llm.groq_client import get_groq_client
            from src.config import GROQ_MODEL

            client = get_groq_client(groq_key)
            messages = [
                {
                    "role": "system",
                    "content": "You are MultiRAG, a helpful AI assistant. For general questions or greetings, respond naturally and friendly. Keep it short.",
                }
            ]
            if chat_history:
                for msg in chat_history[-6:]:
                    messages.append(msg)
            messages.append({"role": "user", "content": query})

            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {"type": "token", "content": chunk.choices[0].delta.content}
        except Exception as e:
            yield {"type": "token", "content": f"Error: {e}"}
        yield {"type": "done", "sources": [], "route": route, "score": 10}
        return

    # Step 3: Detect summary queries — they need more chunks and a special prompt
    _SUMMARY_WORDS = {"summarize", "summary", "overview", "summarise"}
    is_summary = any(w in query.lower() for w in _SUMMARY_WORDS)
    top_k = 30 if is_summary else 10

    yield {"type": "status", "message": "Searching documents..."}
    chunks = _safe_search(query, pinecone_key, top_k=top_k, namespace=namespace)

    if not chunks:
        yield {"type": "token", "content": "No relevant documents found. Please make sure documents are uploaded and try again."}
        yield {"type": "done", "sources": [], "route": route, "score": 0}
        return

    if is_summary:
        # Skip grader for summaries — it would filter out pages not matching the keyword
        # "summarize". Use all retrieved chunks so every section is covered.
        filtered = chunks
        grade = {"score": 8, "filtered_chunks": chunks, "feedback": ""}
    else:
        # Step 4: Grade retrieval
        yield {"type": "status", "message": "Checking relevance..."}
        grade = _safe_grade(query, chunks, groq_key)
        filtered = grade.get("filtered_chunks") or chunks[:5]

        # Step 5: Retry if low quality
        retries = 0
        while grade.get("score", 5) < 4 and retries < MAX_RETRIES:
            retries += 1
            yield {"type": "status", "message": f"Refining search (attempt {retries + 1})..."}
            refined = _safe_refine(query, grade.get("feedback", ""), groq_key)
            chunks = _safe_search(refined, pinecone_key, namespace=namespace)
            if chunks:
                grade = _safe_grade(query, chunks, groq_key)
                filtered = grade.get("filtered_chunks") or chunks[:5]

    # Step 6: Build context
    context = _build_context(filtered)

    # Step 7: Generate answer
    yield {"type": "status", "message": "Generating answer..."}

    # For summaries, prepend a comprehensive instruction so the LLM covers all sections
    if is_summary:
        llm_query = (
            "Provide a comprehensive summary of this document covering ALL major sections "
            "including: company overview, products, technical architecture, HR policies, "
            "customer support, financials, security compliance, and FAQs. "
            + query
        )
    else:
        llm_query = query

    if route == "image" and images:
        try:
            from src.llm.gemini_client import analyze_image
            image_bytes = list(images.values())[0]
            answer = analyze_image(image_bytes, llm_query, google_key)
            yield {"type": "token", "content": answer}
        except Exception as e:
            yield {"type": "token", "content": f"Image analysis error: {e}"}
    else:
        try:
            from src.llm.groq_client import ask_groq_stream
            for token in ask_groq_stream(llm_query, context, chat_history, groq_key):
                yield {"type": "token", "content": token}
        except Exception as e:
            yield {"type": "token", "content": f"Error generating answer: {e}"}

    # Step 8: Return sources
    sources = []
    if filtered:
        for c in filtered[:5]:
            sources.append({
                "page": c.get("page", 0),
                "source": c.get("source", "unknown"),
                "text": c.get("text", "")[:200],
                "score": c.get("score", 0),
            })
    yield {"type": "done", "sources": sources, "route": route, "score": grade.get("score", 5)}


# ──────────────────────────────────────────────
# Safe wrapper functions — never return None
# ──────────────────────────────────────────────

def _safe_search(query: str, pinecone_key: str, top_k: int = 10, namespace: str = "default") -> List[Dict]:
    """Search with error handling. Always returns a list."""
    try:
        from src.vectorstore.pinecone_store import search
        results = search(query, top_k=top_k, api_key=pinecone_key, namespace=namespace)
        if results is None:
            return []
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def _safe_grade(query: str, chunks: List[Dict], groq_key: str) -> Dict:
    """Grade with error handling. Always returns a dict."""
    try:
        from src.agents.grader import grade_retrieval
        result = grade_retrieval(query, chunks, groq_key)
        if result is None:
            return {"is_relevant": True, "score": 5, "filtered_chunks": chunks, "feedback": ""}
        if result.get("filtered_chunks") is None:
            result["filtered_chunks"] = chunks[:5]
        return result
    except Exception as e:
        print(f"Grade error: {e}")
        return {"is_relevant": True, "score": 5, "filtered_chunks": chunks, "feedback": str(e)}


def _safe_refine(query: str, feedback: str, groq_key: str) -> str:
    """Refine query with error handling. Always returns a string."""
    try:
        from src.agents.grader import refine_query
        result = refine_query(query, feedback, groq_key)
        return result if result else query
    except Exception:
        return query


# ──────────────────────────────────────────────
# Handler functions
# ──────────────────────────────────────────────

def _handle_general(query: str, chat_history: list, groq_key: str) -> Dict:
    """Handle general questions without document search."""
    try:
        from src.llm.groq_client import get_groq_client
        from src.config import GROQ_MODEL

        client = get_groq_client(groq_key)
        messages = [
            {
                "role": "system",
                "content": "You are MultiRAG, a helpful AI assistant. For general questions or greetings, respond naturally and friendly. Keep it short.",
            }
        ]
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append(msg)
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": [],
            "route": "general",
            "grade_score": 10,
        }
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "route": "general", "grade_score": 0}


def _handle_image(query: str, images: dict, google_key: str) -> Dict:
    """Handle image questions with Gemini Vision."""
    try:
        from src.llm.gemini_client import analyze_image
        image_bytes = list(images.values())[0]
        answer = analyze_image(image_bytes, query, google_key)
        return {
            "answer": answer,
            "sources": [{"page": 0, "source": "image", "text": "Image analysis", "score": 1.0}],
            "route": "image",
            "grade_score": 8,
        }
    except Exception as e:
        return {"answer": f"Image analysis error: {e}", "sources": [], "route": "image", "grade_score": 0}


def _handle_rag(query: str, chat_history: list, groq_key: str, pinecone_key: str, namespace: str = "default") -> Dict:
    """Handle RAG questions (non-streaming path)."""
    chunks = _safe_search(query, pinecone_key, namespace=namespace)
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": [], "route": "rag", "grade_score": 0}

    grade = _safe_grade(query, chunks, groq_key)
    filtered = grade.get("filtered_chunks") or chunks[:5]

    retries = 0
    while grade.get("score", 5) < 4 and retries < MAX_RETRIES:
        retries += 1
        refined = _safe_refine(query, grade.get("feedback", ""), groq_key)
        chunks = _safe_search(refined, pinecone_key, namespace=namespace)
        if chunks:
            grade = _safe_grade(query, chunks, groq_key)
            filtered = grade.get("filtered_chunks") or chunks[:5]

    context = _build_context(filtered)

    try:
        from src.llm.groq_client import ask_groq
        answer = ask_groq(query, context, chat_history, groq_key)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    sources = []
    if filtered:
        for c in filtered[:5]:
            sources.append({
                "page": c.get("page", 0),
                "source": c.get("source", "unknown"),
                "text": c.get("text", "")[:200],
                "score": c.get("score", 0),
            })

    return {"answer": answer, "sources": sources, "route": "rag", "grade_score": grade.get("score", 5)}


def _build_context(chunks: List[Dict]) -> str:
    """Combine chunks into context string. Never fails."""
    if not chunks:
        return "No relevant information found in the documents."

    parts = []
    for c in chunks:
        if not c or not isinstance(c, dict):
            continue
        text = c.get("text", "")
        source = c.get("source", "unknown")
        page = c.get("page", 0)
        parts.append(f"[Source: {source}, Page {page}]\n{text}")

    if not parts:
        return "No relevant information found in the documents."

    return "\n\n---\n\n".join(parts)
