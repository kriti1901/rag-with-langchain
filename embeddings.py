import os
from langchain_core.embeddings import Embeddings
from config import Config

def get_embeddings() -> Embeddings:
    """
    Returns the appropriate embeddings instance based on the configuration.
    """
    provider = Config.EMBEDDING_PROVIDER
    
    if provider == "local":
        print(f"Initializing local HuggingFace embeddings: {Config.LOCAL_EMBEDDING_MODEL}")
        try:
            # We import here so dependencies are only loaded when needed
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=Config.LOCAL_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"Error loading HuggingFaceEmbeddings: {e}")
            print("Make sure 'sentence-transformers' package is installed.")
            raise e
            
    elif provider == "gemini":
        print("Initializing Google Generative AI embeddings...")
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=Config.GOOGLE_API_KEY
        )
        
    elif provider == "openai":
        print("Initializing OpenAI embeddings...")
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=Config.OPENAI_API_KEY
        )
        
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            "Please choose from 'local', 'gemini', or 'openai'."
        )
