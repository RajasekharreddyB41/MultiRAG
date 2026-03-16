from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL


def get_groq_client(api_key: str = None):
    """Create a Groq client using provided key or default."""
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError("Groq API key is required")
    return Groq(api_key=key)


def ask_groq(question: str, context: str, chat_history: list = None, api_key: str = None):
    """
    Send a question to Groq with retrieved context.
    Returns the answer as a string.
    """
    client = get_groq_client(api_key)

    # Build the system prompt
    system_prompt = """You are a helpful AI assistant for document Q&A.

IMPORTANT — Pronoun resolution: If the question uses "she", "he", "they", "it", \
"her", "him", "his", "their", look at the conversation history to find who or what \
is being referred to, then answer about that specific person/thing. \
For example, if the history shows the CFO is Sarah Chen and the user asks \
"When did she join?", answer about Sarah Chen.

Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information in the provided documents to answer that."
Cite the page number when referencing information."""

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history for conversation memory
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages
            messages.append(msg)

    # Add current question with context
    user_message = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

    messages.append({"role": "user", "content": user_message})

    # Call Groq
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def ask_groq_stream(question: str, context: str, chat_history: list = None, api_key: str = None):
    """
    Same as ask_groq but with streaming for real-time UI.
    Yields tokens one by one.
    """
    client = get_groq_client(api_key)

    system_prompt = """You are a helpful AI assistant for document Q&A.

IMPORTANT — Pronoun resolution: If the question uses "she", "he", "they", "it", \
"her", "him", "his", "their", look at the conversation history to find who or what \
is being referred to, then answer about that specific person/thing. \
For example, if the history shows the CFO is Sarah Chen and the user asks \
"When did she join?", answer about Sarah Chen.

Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information in the provided documents to answer that."
Cite the page number when referencing information."""

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    user_message = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

    messages.append({"role": "user", "content": user_message})

    # Stream the response
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content