import os
import sys
from config import Config
from document import load_document, load_directory, chunk_documents
from embeddings import get_embeddings
from vector_store import load_vector_store, add_documents_to_store, clear_vector_store, is_vector_store_initialized
from rag_pipeline import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

def print_help():
    print("\n--- Production RAG CLI ---")
    print("Commands:")
    print("  ingest <path>  - Ingest a file or a folder of documents")
    print("  query <text>   - Run a quick single query without history")
    print("  chat           - Start an interactive chat session with history")
    print("  clear          - Clear the vector store from disk")
    print("  help           - Show this help message")
    print("  exit           - Exit the application")

def handle_ingest(path: str):
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return

    print(f"Starting ingestion for: {path}")
    try:
        # Load
        if os.path.isdir(path):
            docs = load_directory(path)
        else:
            docs = load_document(path)
            
        if not docs:
            print("No documents were found/loaded.")
            return
            
        # Chunk
        chunks = chunk_documents(docs)
        
        # Embed and Store
        embeddings = get_embeddings()
        add_documents_to_store(chunks, embeddings)
        print("Ingestion completed successfully!")
        
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)

def handle_query(query_text: str):
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)
    if not vector_store:
        print("Error: Vector store is not initialized. Please run 'ingest <path>' first.")
        return
        
    try:
        # Get chain
        chain = get_rag_chain(vector_store)
        print(f"\nQuerying: {query_text}")
        response = chain.invoke({"input": query_text, "chat_history": []})
        print(f"\nAnswer:\n{response['answer']}")
        
        # Print sources
        print("\n--- Sources ---")
        for i, doc in enumerate(response.get("context", [])):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            page_str = f" (Page {page+1})" if page is not None else ""
            print(f"[{i+1}] {os.path.basename(source)}{page_str}")
            
    except Exception as e:
        print(f"Error during query: {e}", file=sys.stderr)

def handle_chat():
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)
    if not vector_store:
        print("Error: Vector store is not initialized. Please run 'ingest <path>' first.")
        return
        
    try:
        chain = get_rag_chain(vector_store)
        chat_history = []
        print("\n--- Chat Started! Type 'exit' to stop chatting ---")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
            if not user_input:
                continue
                
            if user_input.lower() == "exit":
                break
                
            print("\nAssistant is thinking...")
            try:
                response = chain.invoke({"input": user_input, "chat_history": chat_history})
                print(f"\nAssistant:\n{response['answer']}")
                
                # Update chat history
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=response['answer']))
                
                # Keep history to last 10 messages to avoid context overflow
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]
            except Exception as e:
                print(f"Error getting response: {e}")
                
    except Exception as e:
        print(f"Error starting chat: {e}", file=sys.stderr)

def main():
    Config.validate()
    
    if len(sys.argv) > 1:
        # CLI command line arguments
        cmd = sys.argv[1].lower()
        if cmd == "ingest" and len(sys.argv) > 2:
            handle_ingest(sys.argv[2])
        elif cmd == "query" and len(sys.argv) > 2:
            handle_query(" ".join(sys.argv[2:]))
        elif cmd == "chat":
            handle_chat()
        elif cmd == "clear":
            clear_vector_store()
        else:
            print_help()
    else:
        # Interactive CLI Loop
        print("Welcome to the Production RAG CLI!")
        print_help()
        while True:
            try:
                choice = input("\nrag> ").strip().split(maxsplit=1)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
                
            if not choice:
                continue
                
            cmd = choice[0].lower()
            arg = choice[1] if len(choice) > 1 else ""
            
            if cmd == "exit":
                print("Exiting...")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "ingest":
                if not arg:
                    print("Error: Please specify file or directory path. Example: ingest sample.txt")
                else:
                    handle_ingest(arg)
            elif cmd == "query":
                if not arg:
                    print("Error: Please specify query text. Example: query what is langchain?")
                else:
                    handle_query(arg)
            elif cmd == "chat":
                handle_chat()
            elif cmd == "clear":
                clear_vector_store()
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

if __name__ == "__main__":
    main()
