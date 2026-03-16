import google.generativeai as genai
from src.config import GOOGLE_API_KEY, GEMINI_MODEL


def get_gemini_client(api_key: str = None):
    """Configure and return a Gemini model."""
    key = api_key or GOOGLE_API_KEY
    if not key:
        raise ValueError("Google API key is required")
    genai.configure(api_key=key)
    return genai.GenerativeModel(GEMINI_MODEL)


def analyze_image(image_bytes: bytes, question: str, api_key: str = None):
    """
    Send an image to Gemini Vision for analysis.
    Used for charts, diagrams, and scanned pages.
    """
    model = get_gemini_client(api_key)

    prompt = f"""Analyze this image and answer the following question.
Be specific and detailed in your response.

Question: {question}

If this is a chart or graph, describe the data, trends, and key insights.
If this is a table, extract the data in a structured format.
If this is a diagram, explain what it shows."""

    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": image_bytes}]
    )

    return response.text


def describe_image(image_bytes: bytes, api_key: str = None):
    """
    Get a detailed description of an image.
    Used during document ingestion to create text from images.
    """
    model = get_gemini_client(api_key)

    prompt = """Describe this image in detail.
If it's a chart: describe the type, axes, data points, and trends.
If it's a table: extract all data in a structured format.
If it's a diagram: explain all components and connections.
If it's text: transcribe it accurately."""

    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": image_bytes}]
    )

    return response.text


def ask_gemini(question: str, context: str = "", api_key: str = None):
    """
    Ask Gemini a text-only question.
    Used as fallback or for general questions.
    """
    model = get_gemini_client(api_key)

    prompt = f"""Answer this question based on the context provided.
If no context is given, answer from your general knowledge.

Context: {context}

Question: {question}"""

    response = model.generate_content(prompt)

    return response.text