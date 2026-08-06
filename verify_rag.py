import os
import sys

def verify():
    print("=== Verification Script for RAG Pipeline ===")
    
    # 1. Test config imports
    try:
        from config import Config
        print("[OK] config module imported successfully.")
    except Exception as e:
        print(f"[FAIL] config module import failed: {e}")
        return False
        
    # 2. Test document loader
    try:
        from document import chunk_documents
        from langchain_core.documents import Document
        test_doc = Document(page_content="LangChain is a framework for developing LLM applications.", metadata={"source": "test.txt"})
        chunks = chunk_documents([test_doc], chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 0, "No chunks generated"
        print(f"[OK] document module & chunking works (created {len(chunks)} chunk(s)).")
    except Exception as e:
        print(f"[FAIL] document loader verification failed: {e}")
        return False
        
    # 3. Test embeddings import and local embedding init
    try:
        from embeddings import get_embeddings
        # Temporarily force local embedding settings for testing if not set
        Config.EMBEDDING_PROVIDER = "local"
        embeddings = get_embeddings()
        print("[OK] Local HuggingFace embeddings class loaded.")
        
        # Test basic embedding generation
        print("Generating a test embedding vector...")
        vector = embeddings.embed_query("test query")
        assert len(vector) > 0, "Embedding vector is empty"
        print(f"[OK] Embeddings generated successfully! (dimensions: {len(vector)})")
    except Exception as e:
        print(f"[FAIL] Embeddings initialization failed: {e}")
        print("Tip: If sentence-transformers is still installing, please wait.")
        return False
        
    # 4. Test FAISS vector store
    try:
        from vector_store import add_documents_to_store, load_vector_store, clear_vector_store
        print("Initializing clean local vector store...")
        clear_vector_store()
        db = add_documents_to_store(chunks, embeddings)
        assert db is not None, "Vector store returned None"
        
        # Search check
        results = db.similarity_search("LangChain", k=1)
        assert len(results) > 0, "No search results returned"
        print(f"[OK] FAISS Vector store created, saved, and queried. Found: '{results[0].page_content}'")
        
        # Clean up
        clear_vector_store()
        print("[OK] Cleaned up verification database.")
    except Exception as e:
        print(f"[FAIL] FAISS vector store integration failed: {e}")
        return False
        
    print("\n[SUCCESS] All local components (Config, Document, Embeddings, FAISS) verified!")
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
