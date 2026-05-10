import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Costanti
DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text" # Il modello scaricato prima

def ingest_documents():
    print("Inizio il processo di ingestion...")
    
    # 1. Carica tutti i PDF dalla cartella data/
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("Nessun file PDF trovato nella cartella data/")
        return

    documents = []
    for file in pdf_files:
        file_path = os.path.join(DATA_FOLDER, file)
        print(f"Caricamento di: {file}...")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        
    print(f"Caricate {len(documents)} pagine in totale.")

    # 2. Suddividi il testo in blocchi (Chunks)
    # 1000 caratteri per blocco, con 200 di overlap per non tagliare concetti a metà
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documenti suddivisi in {len(chunks)} blocchi (chunks).")

    # 3. Crea gli embeddings usando Ollama in locale
    print(f"Generazione vettori con Ollama ({EMBEDDING_MODEL}) e salvataggio su ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Crea e salva il database vettoriale su disco
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_FOLDER
    )
    
    # Forza il salvataggio su disco (utile per alcune versioni di Chroma)
    db.persist()
    print("Ingestion completata con successo! Il database è pronto in 'chroma_db/'.")

if __name__ == "__main__":
    # Assicurati che la cartella data esista
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Cartella '{DATA_FOLDER}' creata. Inserisci i PDF e riavvia.")
    else:
        ingest_documents()