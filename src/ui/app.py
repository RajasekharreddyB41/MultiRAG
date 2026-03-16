import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import tempfile

# ──────────────────────────────────────────────
# Page Config — MUST be the first Streamlit call
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MultiRAG",
    page_icon="🧠",
    layout="wide",
)

# ──────────────────────────────────────────────
# Safe imports with error catching
# ──────────────────────────────────────────────
try:
    from src.ingestion.pdf_processor import process_pdf
    from src.ingestion.chunker import chunk_documents
    from src.vectorstore.pinecone_store import upsert_chunks, delete_all
    from src.agents.pipeline import run_pipeline_stream
    from src.llm.gemini_client import describe_image
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ──────────────────────────────────────────────
# Show import error if any
# ──────────────────────────────────────────────
if not IMPORTS_OK:
    st.error(f"❌ Import Error: {IMPORT_ERROR}")
    st.info("Check your terminal for details. Some module may have an issue.")
    st.stop()

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False
if "document_images" not in st.session_state:
    st.session_state.document_images = {}
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {"groq": "", "google": "", "pinecone": ""}
if "keys_valid" not in st.session_state:
    st.session_state.keys_valid = False
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("🧠 MultiRAG")
st.sidebar.caption("Multimodal Agentic RAG System")

# API Keys
st.sidebar.subheader("🔑 API Keys")

groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Get free at console.groq.com")
google_key = st.sidebar.text_input("Google API Key", type="password", help="Get free at aistudio.google.com")
pinecone_key = st.sidebar.text_input("Pinecone API Key", type="password", help="Get free at app.pinecone.io")

if st.sidebar.button("✅ Connect", use_container_width=True):
    if groq_key and google_key and pinecone_key:
        st.session_state.api_keys = {
            "groq": groq_key,
            "google": google_key,
            "pinecone": pinecone_key,
        }
        st.session_state.keys_valid = True
        st.sidebar.success("All APIs connected!")
    else:
        st.sidebar.error("Enter all 3 keys")

# Status
if st.session_state.keys_valid:
    st.sidebar.success("✅ APIs connected")
else:
    st.sidebar.warning("⚠️ Enter API keys to start")

st.sidebar.divider()

# Document Upload
st.sidebar.subheader("📄 Upload Documents")

if not st.session_state.keys_valid:
    st.sidebar.info("Connect API keys first to upload")
else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.sidebar.button("🚀 Process Documents", use_container_width=True):
            all_chunks = []
            progress = st.sidebar.progress(0)

            for idx, uploaded_file in enumerate(uploaded_files):
                st.sidebar.text(f"Processing {uploaded_file.name}...")

                # Save to temp with original filename
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(tmp_path, "wb") as tmp:
                    tmp.write(uploaded_file.read())

                try:
                    # Extract content
                    result = process_pdf(tmp_path)

                    # Process images with Gemini
                    for img in result["images"]:
                        try:
                            description = describe_image(
                                img["image_bytes"],
                                st.session_state.api_keys["google"],
                            )
                            result["text_pages"].append({
                                "page": img["page"],
                                "text": f"[Image on page {img['page']}]: {description}",
                                "type": "image",
                                "source": img["source"],
                            })
                            key = f"{img['source']}_p{img['page']}_i{img['image_index']}"
                            st.session_state.document_images[key] = img["image_bytes"]
                        except Exception as e:
                            st.sidebar.warning(f"Image error p{img['page']}: {e}")

                    # Chunk
                    all_pages = result["text_pages"] + result["tables"]
                    chunks = chunk_documents(all_pages)
                    all_chunks.extend(chunks)
                    st.sidebar.text(f"✅ {uploaded_file.name}: {len(chunks)} chunks")

                except Exception as e:
                    st.sidebar.error(f"Error: {uploaded_file.name}: {e}")

                finally:
                    os.unlink(tmp_path)

                progress.progress((idx + 1) / len(uploaded_files))

            # Upload to Pinecone
            if all_chunks:
                st.sidebar.text("Uploading to Pinecone...")
                try:
                    count = upsert_chunks(all_chunks, st.session_state.api_keys["pinecone"])
                    st.session_state.documents_uploaded = True
                    st.session_state.uploaded_files_list = [f.name for f in uploaded_files]
                    st.sidebar.success(f"✅ {count} chunks uploaded!")
                except Exception as e:
                    st.sidebar.error(f"Pinecone error: {e}")

# Show loaded docs
if st.session_state.uploaded_files_list:
    st.sidebar.divider()
    st.sidebar.subheader("📚 Loaded Documents")
    for fname in st.session_state.uploaded_files_list:
        st.sidebar.text(f"• {fname}")

    if st.sidebar.button("🗑️ Clear All", use_container_width=True):
        try:
            delete_all(st.session_state.api_keys["pinecone"])
        except Exception:
            pass
        st.session_state.documents_uploaded = False
        st.session_state.uploaded_files_list = []
        st.session_state.document_images = {}
        st.session_state.messages = []
        st.rerun()

st.sidebar.divider()
st.sidebar.caption("🔒 Keys stored in browser session only")

# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
st.title("🧠 MultiRAG")
st.caption("Multimodal Agentic RAG — Upload documents, ask anything")

# Status messages
if not st.session_state.keys_valid:
    st.info("👈 Enter your API keys in the sidebar to get started")
    st.stop()

if not st.session_state.documents_uploaded:
    st.info("👈 Upload a PDF document in the sidebar to start chatting")

    # Show example of what MultiRAG can do
    st.markdown("### What MultiRAG can do:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📄 **Text Q&A** — Ask about document content")
        st.markdown("📊 **Chart Analysis** — Understand charts & graphs")
    with col2:
        st.markdown("📋 **Table Extraction** — Read tables from PDFs")
        st.markdown("💬 **Follow-ups** — Remembers conversation context")
    st.stop()

# Example questions
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    col1, col2 = st.columns(2)

    examples = [
        "Summarize this document",
        "What are the key insights?",
        "Explain the data in the tables",
        "What is the main conclusion?",
    ]

    for i, example in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(example, use_container_width=True, key=f"ex_{i}"):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander(f"📄 Sources ({len(message['sources'])} references)"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']}** — Page {source['page']} "
                        f"(relevance: {source['score']:.2f})"
                    )
                    st.caption(source["text"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        status_box = st.empty()
        response_box = st.empty()

        full_response = ""
        sources = []

        # Build chat history for the pipeline.
        # - Exclude the just-appended current user message (messages[:-1]) so it
        #   isn't duplicated when the pipeline appends it again with context.
        # - Include BOTH user and assistant messages so the LLM can resolve
        #   follow-up pronouns ("she" → "Sarah Chen") from prior answers.
        # - Skip assistant messages with empty content (e.g. after an error).
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m.get("content", "").strip()
        ][-6:]

        try:
            for event in run_pipeline_stream(
                query=prompt,
                chat_history=chat_history,
                has_documents=st.session_state.documents_uploaded,
                images=st.session_state.document_images,
                api_keys=st.session_state.api_keys,
            ):
                if event["type"] == "status":
                    status_box.caption(f"⏳ {event['message']}")

                elif event["type"] == "token":
                    full_response += event["content"]
                    response_box.markdown(full_response + "▌")

                elif event["type"] == "done":
                    sources = event.get("sources", [])
                    status_box.empty()

            response_box.markdown(full_response)

            if sources:
                with st.expander(f"📄 Sources ({len(sources)} references)"):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']}** — Page {source['page']} "
                            f"(relevance: {source['score']:.2f})"
                        )
                        st.caption(source["text"])
                        st.divider()

        except Exception as e:
            st.error(f"Error generating response: {e}")
            full_response = f"Sorry, an error occurred: {e}"

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

