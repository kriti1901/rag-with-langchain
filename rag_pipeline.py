from typing import Any, Dict
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from config import Config

def get_llm(provider: str = None) -> BaseChatModel:
    """
    Returns the appropriate LLM instance based on the configuration.
    """
    if provider is None:
        provider = Config.LLM_PROVIDER
        
    if provider == "gemini":
        print(f"Initializing Gemini LLM: {Config.GEMINI_MODEL}")
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini LLM.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=Config.TEMPERATURE
        )
        
    elif provider == "openai":
        print(f"Initializing OpenAI LLM: {Config.OPENAI_MODEL}")
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI LLM.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=Config.TEMPERATURE
        )
        
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Please choose from 'gemini' or 'openai'."
        )

def get_rag_chain(vector_store: VectorStore, provider: str = None) -> Any:
    """
    Constructs a history-aware conversational RAG chain.
    """
    llm = get_llm(provider)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 1. Prompt to contextualize/reformulate the user question based on history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 2. Main Question Answering prompt
    system_prompt = (
        "You are a helpful, professional assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer detailed, clear, and accurate. Cite sources or document names "
        "from the context metadata if possible.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create the chain to combine/stuff retrieved documents into the context
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create the final retrieval chain combining retriever & QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
