import html

import streamlit as st

from config import EMBEDDING_MODEL, K_RESULTS, LLM_MODEL, SAMPLE_DOCUMENTS
from documents import (
    index_exists,
    list_loaded_documents,
    rebuild_vector_index,
    remove_document,
    save_uploaded_documents,
)
from styles import APP_CSS


EXAMPLE_PROMPTS = [
    "Summarize the key financial risks.",
    "What macroeconomic assumptions are mentioned?",
    "List the most important figures and explain their context.",
]


def configure_page():
    st.set_page_config(
        page_title="Financial RAG Assistant",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_document_card(document):
    document_name = html.escape(document["name"])
    document_size = html.escape(document["size"])
    sample_badge = '<div class="doc-badge">Bundled sample</div>' if document["is_sample"] else ""
    st.markdown(
        f"""
        <div class="doc-row">
            <div class="doc-name">{document_name}</div>
            <div class="doc-meta">{document_size} - PDF</div>
            {sample_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(on_index_rebuilt):
    documents = list_loaded_documents()
    db_ready = index_exists()

    with st.sidebar:
        st.markdown("## Financial RAG")
        status_class = "status-ready" if db_ready else "status-missing"
        status_text = "Index ready" if db_ready else "Index missing"
        st.markdown(
            f'<span class="status-pill {status_class}">{status_text}</span>',
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        col_a.metric("PDFs", len(documents))
        col_b.metric("Chunks", K_RESULTS)

        st.divider()
        st.markdown("### Documents")

        if documents:
            for document in documents:
                render_document_card(document)
        else:
            st.info("No PDF files found in data/.")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Uploaded files are saved in the local document folder.",
        )
        if uploaded_files and st.button("Add uploaded PDFs", use_container_width=True):
            saved_files = save_uploaded_documents(uploaded_files)
            if saved_files:
                st.success(f"Added {len(saved_files)} PDF file(s). Rebuild the index to include them.")
                st.rerun()
            else:
                st.warning("No valid PDF files were uploaded.")

        if documents:
            removable_names = [document["name"] for document in documents]
            selected_document = st.selectbox("Remove a PDF", removable_names)
            if selected_document in SAMPLE_DOCUMENTS:
                st.caption("This is the bundled sample. Removing it only deletes your local copy.")

            if st.button("Remove selected PDF", use_container_width=True):
                if remove_document(selected_document):
                    st.warning(f"Removed {selected_document}. Rebuild the index to refresh search results.")
                    st.rerun()
                else:
                    st.error("The selected file could not be removed.")

        st.divider()
        if st.button("Rebuild document index", type="primary", use_container_width=True):
            with st.spinner("Rebuilding the vector index..."):
                try:
                    file_count, chunk_count = rebuild_vector_index()
                    on_index_rebuilt()
                    st.success(f"Indexed {file_count} PDF file(s) into {chunk_count} chunks.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Index rebuild failed: {exc}")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

        st.caption(f"LLM: {LLM_MODEL}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")


def render_header():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="app-kicker">Local financial document intelligence</div>
            <div class="app-title">Financial RAG Assistant</div>
            <div class="app-subtitle">
                Upload financial PDFs, rebuild the local index, and ask grounded questions over your private document set.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False


def render_quick_prompts():
    prompt_cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, example in zip(prompt_cols, EXAMPLE_PROMPTS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_current_prompt():
    chat_prompt = st.chat_input("Example: What are the main risk factors mentioned?")
    prompt = st.session_state.pending_prompt or chat_prompt
    st.session_state.pending_prompt = None
    return prompt


def render_chat_response(qa_chain, prompt):
    with st.chat_message("assistant"):
        if qa_chain is None:
            response = "The document index is not ready yet. Upload one or more PDFs, then click **Rebuild document index** in the sidebar."
            st.warning(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return

        stop_requested = st.button("Stop generation", key="stop_generation", use_container_width=False)
        if stop_requested:
            st.session_state.is_generating = False
            st.info("Generation stopped.")
            st.stop()

        response_placeholder = st.empty()
        response = ""
        st.session_state.is_generating = True

        try:
            with st.spinner("Searching the financial documents..."):
                for chunk in qa_chain.stream(prompt):
                    response += chunk
                    response_placeholder.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as exc:
            st.error(f"An error occurred: {exc}")
        finally:
            st.session_state.is_generating = False


def handle_chat(qa_chain):
    render_quick_prompts()
    render_chat_history()

    prompt = get_current_prompt()
    if not prompt:
        st.session_state.is_generating = False
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    render_chat_response(qa_chain, prompt)
