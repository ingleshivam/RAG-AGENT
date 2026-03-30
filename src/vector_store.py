import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

COLLECTION_NAME = "rag_documents"

def get_qdrant_client():
    if QDRANT_URL and QDRANT_API_KEY:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    else:
        # Fallback to local persistent if cloud not configured
        client = QdrantClient(path="data/qdrant_local")
        
    return client

def ensure_payload_index(client):
    """
    Creates a payload index for metadata.document_name to allow efficient filtering.
    """
    try:
        # Check if collection exists first to avoid error if it's not created yet
        collections = client.get_collections().collections
        if any(c.name == COLLECTION_NAME for c in collections):
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.document_name",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"Payload index ensured for {COLLECTION_NAME}")
    except Exception as e:
        print(f"Warning: Could not create payload index: {e}")

def get_embeddings_model():
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text:v1.5"
    )

def parse_extracted_text(file_path: str):
    """
    Parses the text file formatted by pdf_processor.py
    and returns a list of LangChain Document objects with page metadata.
    """
    doc_name = os.path.basename(file_path).replace(".txt", ".pdf")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split by standard page marker: --- PAGE X ---
    pages = re.split(r'--- PAGE (\d+) ---', content)
    
    documents = []
    # pages[0] is text before the first page (usually empty)
    for i in range(1, len(pages), 2):
        page_num = int(pages[i])
        page_text = pages[i+1].strip()
        
        if page_text:
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"document_name": doc_name, "page_number": page_num}
                )
            )
            
    return documents

def get_vector_store():
    """
    Returns an initialized QdrantVectorStore wrapper. Expected to be used during retrieval.
    """
    client = get_qdrant_client()
    ensure_payload_index(client) # Ensure index for filtering
    embeddings = get_embeddings_model()
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store

def store_documents_in_qdrant(text_files):
    """
    Chunks documents, embeds, and stores them in Qdrant.
    """
    embeddings = get_embeddings_model()
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for file_path in text_files:
        docs = parse_extracted_text(file_path)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)
        
    print(f"Total chunks to insert: {len(all_chunks)}")
    
    if all_chunks:
        if QDRANT_URL and QDRANT_API_KEY:
            vector_store = QdrantVectorStore.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                force_recreate=False
            )
        else:
            vector_store = QdrantVectorStore.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                path="data/qdrant_local",
                force_recreate=False
            )
        return vector_store
    return None

if __name__ == "__main__":
    import glob
    os.makedirs("data/qdrant_local", exist_ok=True)
    txt_files = glob.glob("data/extracted_text/*.txt")
    if txt_files:
        store_documents_in_qdrant(txt_files)
        print("Embeddings stored successfully.")
    else:
        print("No text files found in data/extracted_text")
