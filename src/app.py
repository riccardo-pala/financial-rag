import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os

# --- COSTANTI & PERCORSI ---
# Trova la cartella 'src' e poi sale di un livello fino alla root del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_FOLDER = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# --- CONFIGURAZIONE INTERFACCIA ---
st.set_page_config(page_title="Financial RAG Assistant", page_icon="🏦")
st.title("🏦 Financial RAG Assistant")
st.markdown("Chiedimi qualsiasi cosa sui documenti finanziari che hai caricato!")

# --- INIZIALIZZAZIONE SISTEMA RAG ---
# Usiamo st.cache_resource così carica il database e il modello solo la prima volta
@st.cache_resource
def load_rag_chain():
    # 1. Ricarica il database vettoriale
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    
    # 2. Configura il Retriever (recupera i 4 blocchi di testo più pertinenti)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # 3. Inizializza il Large Language Model (Llama 3)
    llm = Ollama(model=LLM_MODEL)
    
    # 4. Creiamo un Prompt ingegnerizzato (Prompt Engineering)
    # Diciamo al modello di usare SOLO i documenti per evitare "allucinazioni"
    template = """Sei un consulente finanziario esperto. 
    Usa ESCLUSIVAMENTE le seguenti porzioni di contesto recuperato dai documenti per rispondere alla domanda. 
    Se la risposta non è contenuta nel contesto, rispondi: "Mi dispiace, ma non ho trovato informazioni a riguardo nei documenti caricati." Non inventare mai cifre o dati.

    Contesto: {context}

    Domanda: {question}

    Risposta:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # 5. Uniamo tutto in una Catena (Chain) di LangChain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Carica il sistema
qa_chain = load_rag_chain()

# --- GESTIONE DELLA CHAT UI ---
# Inizializza la cronologia della chat nello state di Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
if prompt := st.chat_input("Es: Quali sono i principali fattori di rischio menzionati?"):
    # Salva e mostra il messaggio dell'utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genera la risposta dell'AI
    with st.chat_message("assistant"):
        with st.spinner("Sto consultando i documenti finanziari..."):
            try:
                # Esegui la catena RAG
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]
                
                # Mostra la risposta
                st.markdown(response)
                
                # Salva la risposta nella cronologia
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")