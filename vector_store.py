import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from config import Config

def get_vector_store_path() -> str:
    """Returns the absolute path to the local vector store."""
    return Config.DB_DIR

def is_vector_store_initialized() -> bool:
    """Checks if the local FAISS index exists on disk."""
    db_path = get_vector_store_path()
    index_file = os.path.join(db_path, "index.faiss")
    pkl_file = os.path.join(db_path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(pkl_file)

def load_vector_store(embeddings: Embeddings) -> Optional[FAISS]:
    """Loads the vector store from disk if it exists, otherwise returns None."""
    if is_vector_store_initialized():
        db_path = get_vector_store_path()
        print(f"Loading existing vector store from: {db_path}")
        # allow_dangerous_deserialization is required for loading local pickle files in newer langchain
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    print("Vector store not initialized on disk.")
    return None

def save_vector_store(vector_store: FAISS) -> None:
    """Saves the vector store to disk."""
    db_path = get_vector_store_path()
    os.makedirs(db_path, exist_ok=True)
    vector_store.save_local(db_path)
    print(f"Saved vector store to: {db_path}")

def add_documents_to_store(documents: List[Document], embeddings: Embeddings) -> FAISS:
    """
    Adds a list of LangChain Documents to the vector store.
    If the store exists, it merges/appends the documents.
    If not, it creates a new index.
    """
    if not documents:
        raise ValueError("Cannot add empty list of documents to vector store.")

    existing_store = load_vector_store(embeddings)
    if existing_store:
        print(f"Adding {len(documents)} document chunk(s) to existing vector store...")
        existing_store.add_documents(documents)
        save_vector_store(existing_store)
        return existing_store
    else:
        print(f"Creating new vector store with {len(documents)} document chunk(s)...")
        new_store = FAISS.from_documents(documents, embeddings)
        save_vector_store(new_store)
        return new_store

def clear_vector_store() -> None:
    """Deletes the local FAISS database from disk."""
    db_path = get_vector_store_path()
    index_file = os.path.join(db_path, "index.faiss")
    pkl_file = os.path.join(db_path, "index.pkl")
    
    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
    if os.path.exists(db_path) and not os.listdir(db_path):
        os.rmdir(db_path)
    print("Vector store cleared from disk.")
