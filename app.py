import os
import tempfile
import streamlit as st
from config import Config
from document import load_document, chunk_documents
from embeddings import get_embeddings
from vector_store import load_vector_store, add_documents_to_store, clear_vector_store, is_vector_store_initialized
from rag_pipeline import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# App page config
st.set_page_config(
    page_title="Production RAG Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom premium CSS styling
st.markdown("""
<style>
    /* Main body background and font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Premium glass card styles */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Sources tag style */
    .source-tag {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-ok {
        color: #10b981;
        font-weight: 600;
    }
    .status-warn {
        color: #f59e0b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🌌 Antigravity RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Conversational retrieval over your documents using LangChain, FAISS, and Gemini/OpenAI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
    st.markdown("### ⚙️ Pipeline Configuration")
    
    # Provider Settings (read-only for display / can be updated via UI inputs)
    emb_provider = st.selectbox(
        "Embedding Provider",
        options=["local", "gemini", "openai"],
        index=["local", "gemini", "openai"].index(Config.EMBEDDING_PROVIDER)
    )
    Config.EMBEDDING_PROVIDER = emb_provider
    
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai"],
        index=["gemini", "openai"].index(Config.LLM_PROVIDER)
    )
    Config.LLM_PROVIDER = llm_provider
    
    # Model parameters
    st.markdown("---")
    st.markdown("### 📊 Model Hyperparameters")
    
    chunk_size = st.slider("Chunk Size (characters)", min_value=500, max_value=3000, value=Config.CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=800, value=Config.CHUNK_OVERLAP, step=50)
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=Config.TEMPERATURE, step=0.1)
    
    # Apply parameters back to Config
    Config.CHUNK_SIZE = chunk_size
    Config.CHUNK_OVERLAP = chunk_overlap
    Config.TEMPERATURE = temperature
    
    # Validate API Keys
    st.markdown("---")
    st.markdown("### 🔑 API Key Authentication")
    
    google_key = st.text_input("GOOGLE_API_KEY", value=Config.GOOGLE_API_KEY or "", type="password")
    openai_key = st.text_input("OPENAI_API_KEY", value=Config.OPENAI_API_KEY or "", type="password")
    
    # Update Config keys
    if google_key:
        Config.GOOGLE_API_KEY = google_key
    if openai_key:
        Config.OPENAI_API_KEY = openai_key
        
    # Check setup status
    st.markdown("---")
    st.markdown("### 🛠️ System Health")
    
    # Embeddings health status
    if Config.EMBEDDING_PROVIDER == "local":
        st.markdown("Embeddings: <span class='status-ok'>Local (Active)</span>", unsafe_allow_html=True)
    elif Config.EMBEDDING_PROVIDER == "gemini" and Config.GOOGLE_API_KEY:
        st.markdown("Embeddings: <span class='status-ok'>Gemini (Active)</span>", unsafe_allow_html=True)
    elif Config.EMBEDDING_PROVIDER == "openai" and Config.OPENAI_API_KEY:
        st.markdown("Embeddings: <span class='status-ok'>OpenAI (Active)</span>", unsafe_allow_html=True)
    else:
        st.markdown("Embeddings: <span class='status-warn'>Key Missing</span>", unsafe_allow_html=True)
        
    # LLM health status
    if Config.LLM_PROVIDER == "gemini" and Config.GOOGLE_API_KEY:
        st.markdown("LLM: <span class='status-ok'>Gemini (Active)</span>", unsafe_allow_html=True)
    elif Config.LLM_PROVIDER == "openai" and Config.OPENAI_API_KEY:
        st.markdown("LLM: <span class='status-ok'>OpenAI (Active)</span>", unsafe_allow_html=True)
    else:
        st.markdown("LLM: <span class='status-warn'>Key Missing</span>", unsafe_allow_html=True)

# Main Application split: Left (Upload & DB), Right (Chat UI)
col_upload, col_chat = st.columns([1, 2])

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📥 Upload & Ingest Documents")
    
    uploaded_files = st.file_uploader(
        "Select PDF, TXT, or MD files:",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 Ingest Selected Files", use_container_width=True) and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            temp_docs = []
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Write to temp file to allow loaders to access it
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name
                
                try:
                    # Load document
                    loaded_chunks = load_document(temp_path)
                    
                    # Update source metadata to show original filename
                    for chunk in loaded_chunks:
                        chunk.metadata["source"] = uploaded_file.name
                        
                    temp_docs.extend(loaded_chunks)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                progress_bar.progress(int(((i + 0.5) / len(uploaded_files)) * 50))
            
            # Chunk all documents
            status_text.text("Splitting documents into chunks...")
            chunks = chunk_documents(temp_docs, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
            progress_bar.progress(70)
            
            # Embed & store
            status_text.text("Generating embeddings and writing to FAISS...")
            embeddings = get_embeddings()
            add_documents_to_store(chunks, embeddings)
            
            progress_bar.progress(100)
            status_text.text("")
            st.success(f"Successfully ingested {len(uploaded_files)} files into vector store!")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ingestion failed: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Vector DB Operations Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📁 Vector Database")
    
    db_initialized = is_vector_store_initialized()
    if db_initialized:
        st.info("Vector database status: **Active & Ready**")
        if st.button("🗑️ Clear Vector Database", type="secondary", use_container_width=True):
            clear_vector_store()
            st.warning("Vector store cleared from disk.")
            st.rerun()
    else:
        st.warning("Vector database: **Empty / Not initialized**")
        st.write("Please upload and ingest files to initialize the database.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_chat:
    st.subheader("💬 Chat Interface")
    
    # Initialize chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "langchain_history" not in st.session_state:
        st.session_state.langchain_history = []
        
    # Clear chat button
    if len(st.session_state.messages) > 0:
        if st.button("🔄 Reset Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.langchain_history = []
            st.rerun()
            
    # Render chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown("**Sources:**")
                for src in msg["sources"]:
                    st.markdown(f"<span class='source-tag'>{src}</span>", unsafe_allow_html=True)
                    
    # Handle user query
    if user_query := st.chat_input("Ask a question about your ingested documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Load vector store
        try:
            embeddings = get_embeddings()
            vector_store = load_vector_store(embeddings)
        except Exception as e:
            st.error(f"Initialization error: {e}")
            vector_store = None
            
        if not vector_store:
            with st.chat_message("assistant"):
                st.error("Vector store is not initialized or missing embeddings package. Please ingest documents first.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    # Retrieve RAG chain
                    chain = get_rag_chain(vector_store)
                    
                    # Invoke RAG chain with user query and chat history
                    response = chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.langchain_history
                    })
                    
                    answer = response["answer"]
                    context_docs = response.get("context", [])
                    
                    # Deduplicate and format sources
                    sources_list = []
                    for doc in context_docs:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", None)
                        page_str = f" (Page {page+1})" if page is not None else ""
                        formatted_src = f"{os.path.basename(source)}{page_str}"
                        if formatted_src not in sources_list:
                            sources_list.append(formatted_src)
                            
                    # Display response
                    message_placeholder.markdown(answer)
                    if sources_list:
                        st.markdown("**Sources:**")
                        for src in sources_list:
                            st.markdown(f"<span class='source-tag'>{src}</span>", unsafe_allow_html=True)
                            
                    # Update local Streamlit state and LangChain memory structures
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_list
                    })
                    st.session_state.langchain_history.append(HumanMessage(content=user_query))
                    st.session_state.langchain_history.append(AIMessage(content=answer))
                    
                    # Keep memory size bounded
                    if len(st.session_state.langchain_history) > 10:
                        st.session_state.langchain_history = st.session_state.langchain_history[-10:]
                        
                except Exception as e:
                    message_placeholder.markdown(f"**Error generating response:** {e}")
