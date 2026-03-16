import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()


def get_key(key_name: str) -> str:
    """
    Get API key from Streamlit secrets (production)
    or .env file (local development).
    Returns empty string if not found.
    """
    # Try Streamlit secrets first (for deployed app)
    try:
        value = st.secrets.get(key_name, "")
        if value:
            return value
    except Exception:
        pass

    # Fall back to .env file (for local development)
    return os.getenv(key_name, "")


# API Keys
GROQ_API_KEY = get_key("GROQ_API_KEY")
GOOGLE_API_KEY = get_key("GOOGLE_API_KEY")
PINECONE_API_KEY = get_key("PINECONE_API_KEY")
LANGSMITH_API_KEY = get_key("LANGSMITH_API_KEY")

# Pinecone Settings
PINECONE_INDEX_NAME = get_key("PINECONE_INDEX_NAME") or "multirag-index"

# Model Settings
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking Settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 75

# LangSmith Tracing
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "MultiRAG"