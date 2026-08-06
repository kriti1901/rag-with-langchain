import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from config import Config

def load_document(file_path: str) -> List[Document]:
    """
    Loads a document based on its file extension.
    Supported extensions: .txt, .md, .pdf
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())
    
    if ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Validate that we actually extracted text
        total_text_len = sum(len(doc.page_content.strip()) for doc in docs)
        if total_text_len == 0:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "This usually happens if the PDF is scanned (contains images/scans of text rather than selectable text) "
                "or if it is encrypted."
            )
        return docs
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_directory(directory_path: str, extensions: List[str] = [".txt", ".md", ".pdf"]) -> List[Document]:
    """
    Scans a directory and loads all supported documents.
    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in extensions:
                full_path = os.path.join(root, file)
                print(f"Loading {full_path}...")
                try:
                    documents.extend(load_document(full_path))
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
    return documents

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = None, 
    chunk_overlap: int = None
) -> List[Document]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = Config.CHUNK_OVERLAP
        
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s).")
    return chunks
