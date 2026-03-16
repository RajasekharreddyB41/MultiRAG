# 🧠 MultiRAG

**Multimodal Agentic RAG System — Upload documents, ask anything.**

MultiRAG is a production-grade Retrieval-Augmented Generation system that processes PDFs containing text, images, tables, and charts. It uses an agentic pipeline with intelligent query routing, self-correcting retrieval, hybrid search, and streaming responses.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multirag-rajasekhar.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎬 Demo

> Upload any PDF → Ask questions → Get cited answers with source references

**Live Demo:** [multirag-rajasekhar.streamlit.app](https://multirag-rajasekhar.streamlit.app)

*Bring your own API keys (free tiers available for all services)*

---

## ✨ Features

### Core RAG Engine
- **Multimodal Document Processing** — Extracts text, images, tables, and charts from PDFs using PyMuPDF, Camelot, and Tesseract OCR
- **Hybrid Search** — Combines semantic vector search (Pinecone) with keyword matching for superior retrieval accuracy
- **Smart Chunking** — Full-page chunks preserve tables and lists; smaller overlapping chunks enable precise retrieval
- **Source Citations** — Every answer includes page numbers and expandable source text viewers

### Agentic Pipeline
- **Query Router Agent** — Automatically classifies questions into text RAG, image analysis, table extraction, or general knowledge
- **Self-Correcting Retrieval** — Grades retrieval quality (0-10) and automatically refines the search query if results are poor (max 2 retries)
- **Hallucination Guard** — Refuses to answer when retrieved context doesn't contain relevant information instead of making things up

### Production Features
- **Streaming Responses** — Token-by-token streaming via Groq for a ChatGPT-like experience
- **Conversation Memory** — Maintains chat history for follow-up questions
- **BYOK (Bring Your Own Key)** — Users enter their own API keys; keys stored in browser session only, never on servers
- **Example Questions** — Clickable starter questions to guide new users
- **Document Management** — Upload multiple PDFs, view loaded documents, clear and reset

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Query Router │ ── General ──▶ Groq LLM (direct answer)
│   Agent      │ ── Image ───▶ Gemini Vision
└──────┬───────┘ ── Table ───▶ Table Extraction
       │ RAG
       ▼
┌──────────────┐
│ Hybrid Search│ ── Vector Search (Pinecone)
│              │ ── Keyword Search (local)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Retrieval   │ ── Score < 4? ──▶ Refine query & retry
│   Grader     │ ── Score ≥ 4? ──▶ Continue
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Groq LLM   │ ── Streaming response with citations
│  Generation  │
└──────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Text LLM | Groq (LLaMA 3.3 70B) | Free |
| Vision LLM | Google Gemini 2.0 Flash | Free |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free (local) |
| Vector Database | Pinecone (Serverless) | Free tier |
| PDF Processing | PyMuPDF + Camelot + Tesseract | Free |
| Agent Framework | LangGraph + LangChain | Free |
| Frontend | Streamlit | Free |
| Deployment | Streamlit Cloud | Free |
| Tracing | LangSmith | Free tier |
| **Total** | | **$0** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Free API keys from: [Groq](https://console.groq.com), [Google AI Studio](https://aistudio.google.com/apikey), [Pinecone](https://app.pinecone.io)



## 📁 Project Structure

```
MultiRAG/
├── src/
│   ├── agents/
│   │   ├── router.py          # Query classification (RAG/image/table/general)
│   │   ├── grader.py          # Retrieval quality scoring & self-correction
│   │   └── pipeline.py        # Main orchestrator connecting all components
│   ├── embeddings/
│   │   └── embedder.py        # HuggingFace sentence-transformers
│   ├── ingestion/
│   │   ├── pdf_processor.py   # PDF text, image, table extraction
│   │   └── chunker.py         # Smart text chunking with full-page preservation
│   ├── llm/
│   │   ├── groq_client.py     # Groq API (text generation + streaming)
│   │   └── gemini_client.py   # Google Gemini (vision + text)
│   ├── vectorstore/
│   │   └── pinecone_store.py  # Pinecone CRUD + hybrid search
│   ├── ui/
│   │   └── app.py             # Streamlit frontend
│   └── config.py              # Centralized configuration
├── .env                        # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔑 API Keys Guide

All API keys are free. Here's how to get them:

| Key | Where | Steps | Time |
|-----|-------|-------|------|
| Groq | [console.groq.com](https://console.groq.com) | Sign up → API Keys → Create | 1 min |
| Google | [aistudio.google.com](https://aistudio.google.com/apikey) | Sign up → Create API Key | 1 min |
| Pinecone | [app.pinecone.io](https://app.pinecone.io) | Sign up → API Keys → Copy | 1 min |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) | Sign up → Settings → API Keys | 1 min |

**Security:** Keys entered in the app are stored in your browser session only. They are never saved on any server or transmitted to third parties.





---

## 👤 Author

**Rajasekhar Reddy B**

- GitHub: [@RajasekharreddyB41](https://github.com/RajasekharreddyB41)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Google Gemini](https://ai.google.dev) — Multimodal AI capabilities
- [Pinecone](https://pinecone.io) — Vector database
- [LangChain](https://langchain.com) / [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent framework
- [Streamlit](https://streamlit.io) — Frontend framework
- [HuggingFace](https://huggingface.co) — Open-source embeddings
