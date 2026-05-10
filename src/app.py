import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM

import os
import html
from pathlib import Path

# --- CONSTANTS & PATHS ---
# Find the src folder, then move one level up to the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_FOLDER = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
K_RESULTS = 4

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 980px;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #172033 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff;
    }
    .app-kicker {
        color: #64748b;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }
    .app-title {
        color: #0f172a;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0;
        line-height: 1.08;
        margin-bottom: 0.35rem;
    }
    .app-subtitle {
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.4rem;
    }
    .doc-row {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        margin-bottom: 0.45rem;
        padding: 0.65rem 0.7rem;
    }
    .doc-name {
        color: #ffffff;
        font-size: 0.88rem;
        font-weight: 700;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .doc-meta {
        color: #cbd5e1;
        font-size: 0.76rem;
        margin-top: 0.22rem;
    }
    .status-pill {
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        padding: 0.28rem 0.65rem;
    }
    .status-ready {
        background: #dcfce7;
        color: #166534;
    }
    .status-missing {
        background: #fee2e2;
        color: #991b1b;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_loaded_documents():
    data_path = Path(DATA_FOLDER)
    if not data_path.exists():
        return []

    documents = []
    for file_path in sorted(data_path.glob("*.pdf")):
        documents.append(
            {
                "name": file_path.name,
                "size": format_file_size(file_path.stat().st_size),
            }
        )
    return documents


def render_sidebar():
    documents = list_loaded_documents()
    db_ready = Path(DB_FOLDER, "chroma.sqlite3").exists()

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
                document_name = html.escape(document["name"])
                document_size = html.escape(document["size"])
                st.markdown(
                    f"""
                    <div class="doc-row">
                        <div class="doc-name">{document_name}</div>
                        <div class="doc-meta">{document_size} - PDF</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No PDF files found in data/.")

        st.divider()
        st.caption(f"LLM: {LLM_MODEL}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")


render_sidebar()

st.markdown('<div class="app-kicker">Local financial document intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Financial RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask grounded questions over your indexed financial PDFs. Answers stay tied to the local document context.</div>',
    unsafe_allow_html=True,
)

# --- RAG SYSTEM INITIALIZATION ---
# st.cache_resource loads the database and model only once.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_rag_chain():
    # 1. Reload the vector database.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    
    # 2. Configure the retriever.
    retriever = db.as_retriever(search_kwargs={"k": K_RESULTS})
    
    # 3. Initialize the large language model.
    llm = OllamaLLM(model=LLM_MODEL)
    
    # 4. Build a strict prompt to keep answers grounded in the retrieved documents.
    template = """You are an expert financial advisor.
    Use ONLY the following context excerpts retrieved from the documents to answer the question.
    If the answer is not contained in the context, answer: "I am sorry, but I could not find information about that in the loaded documents." Never invent figures or data.

    Context: {context}

    Question: {question}

    Answer:"""
    
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # 5. Combine retrieval, prompt, and model in a RAG pipeline.
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return qa_chain

# Load the system.
qa_chain = load_rag_chain()

# --- CHAT UI HANDLING ---
# Initialize the chat history in Streamlit session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

with st.sidebar:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

example_prompts = [
    "Summarize the key financial risks.",
    "What macroeconomic assumptions are mentioned?",
    "List the most important figures and explain their context.",
]

with st.container():
    prompt_cols = st.columns(len(example_prompts))
    for col, example in zip(prompt_cols, example_prompts):
        if col.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input.
chat_prompt = st.chat_input("Example: What are the main risk factors mentioned?")
prompt = st.session_state.pending_prompt or chat_prompt
if prompt:
    st.session_state.pending_prompt = None

    # Save and display the user message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the AI response.
    with st.chat_message("assistant"):
        with st.spinner("Searching the financial documents..."):
            try:
                # Run the RAG chain.
                response = qa_chain.invoke(prompt)
                
                # Display the response.
                st.markdown(response)
                
                # Save the response in the chat history.
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
