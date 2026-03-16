from src.llm.groq_client import get_groq_client
from src.config import GROQ_MODEL


# Keywords that suggest image/chart questions
IMAGE_KEYWORDS = [
    "chart", "graph", "diagram", "image", "picture", "figure",
    "plot", "visual", "illustration", "screenshot", "photo",
    "bar chart", "pie chart", "line graph", "flow chart",
]

# Keywords that suggest table questions
TABLE_KEYWORDS = [
    "table", "row", "column", "cell", "spreadsheet",
    "data table", "comparison table", "list of",
]

# Greetings and casual messages — skip document search
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "how are you", "what's up", "sup",
    "thanks", "thank you", "bye", "goodbye", "see you",
    "who are you", "what can you do", "help",
]

# Query prefixes that signal a general knowledge / concept question
_GENERAL_PREFIXES = (
    "what is ", "what are ", "what does ", "what do ",
    "explain ", "define ", "how does ", "how do ", "tell me about ",
)

# Words that anchor a query to the uploaded document — if present, don't route to general
_DOC_WORDS = {
    "novatech", "cloudmind", "shieldguard", "nexusconnect",
    "company", "employee", "policy", "document", "pdf", "report",
    "page", "section", "uploaded", "file", "this", "the document",
    "our", "your", "their", "its",
}


def route_query(query: str, has_documents: bool = True, api_key: str = None) -> str:
    """
    Decide how to handle the query.
    Returns one of: 'rag', 'image', 'table', 'general'
    """
    query_lower = query.lower().strip()

    # Check greetings FIRST — before anything else
    for greeting in GREETING_KEYWORDS:
        if query_lower == greeting or query_lower.startswith(greeting + " ") or query_lower.startswith(greeting + ","):
            return "general"

    # General knowledge check: "what is X", "explain X", "define X", etc.
    # Route to general UNLESS the query contains a doc-anchoring word.
    is_general_prefix = any(query_lower.startswith(p) for p in _GENERAL_PREFIXES)
    has_doc_word = any(w in query_lower for w in _DOC_WORDS)
    if is_general_prefix and not has_doc_word:
        print(f"[router] general-knowledge route: '{query}'")
        return "general"

    # Very short queries (1-3 words) that aren't specific are general
    if len(query_lower.split()) <= 2 and not has_documents:
        return "general"

    # If no documents uploaded, everything is general
    if not has_documents:
        return "general"

    # Check for image-related questions
    for keyword in IMAGE_KEYWORDS:
        if keyword in query_lower:
            return "image"

    # Check for table-related questions
    for keyword in TABLE_KEYWORDS:
        if keyword in query_lower:
            return "table"

    # Use LLM to classify ambiguous queries
    try:
        client = get_groq_client(api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a query classifier. Classify the user's question into exactly ONE category.

Respond with ONLY one word:
- "rag" if the question is about document content, facts, policies, data, or specific information
- "image" if the question asks about charts, graphs, diagrams, or images
- "table" if the question asks about tables, rows, columns, or structured data
- "general" if the question is casual, a greeting, off-topic, or not about any document

Respond with ONLY the category word, nothing else.""",
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=10,
        )

        route = response.choices[0].message.content.strip().lower()

        if route in ["rag", "image", "table", "general"]:
            return route

    except Exception:
        pass

    # Default to RAG if classification fails
    return "rag"


def get_route_description(route: str) -> str:
    """Human-readable description of the route taken."""
    descriptions = {
        "rag": "Searching documents for relevant information...",
        "image": "Analyzing image/chart with Gemini Vision...",
        "table": "Extracting table data from documents...",
        "general": "Answering from general knowledge...",
    }
    return descriptions.get(route, "Processing your question...")