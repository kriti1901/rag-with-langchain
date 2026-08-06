import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class Config:
    # Directory settings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(BASE_DIR, "vectorstore")
    
    # Text splitting settings
    CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    
    # Embeddings settings
    # Options: "local" (HuggingFace), "openai", "gemini"
    EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "local").lower()
    LOCAL_EMBEDDING_MODEL = os.getenv("RAG_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # LLM settings
    # Options: "gemini", "openai"
    LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "gemini").lower()
    
    # Model names
    GEMINI_MODEL = os.getenv("RAG_GEMINI_MODEL", "gemini-3.5-flash")
    OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
    
    # Generation parameters
    TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def validate(cls):
        """Validates that necessary API keys are present based on provider choices."""
        if cls.EMBEDDING_PROVIDER == "gemini" or cls.LLM_PROVIDER == "gemini":
            if not cls.GOOGLE_API_KEY:
                print("Warning: GOOGLE_API_KEY is not set. Gemini services will fail.")
                
        if cls.EMBEDDING_PROVIDER == "openai" or cls.LLM_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                print("Warning: OPENAI_API_KEY is not set. OpenAI services will fail.")
