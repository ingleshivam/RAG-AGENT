import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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

def get_sparse_embeddings_model():
    """Returns a BM25 sparse embedding model via FastEmbed for keyword-based retrieval."""
    return FastEmbedSparse(model_name="Qdrant/bm25")

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
    Returns an initialized QdrantVectorStore wrapper with hybrid (dense + sparse) support.
    """
    client = get_qdrant_client()
    ensure_payload_index(client)
    embeddings = get_embeddings_model()
    sparse_embeddings = get_sparse_embeddings_model()
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    return vector_store

def get_summarizer_llm():
    return Groq()

def summarize_chunk(chunk_text: str, client: Groq) -> str:
    prompt = (
        "Summarize the following text objectively and accurately in one concise natural language sentence. "
        "Focus on the core content. Do not include any introductory phrases like 'This text is about'.\n\n"
        f"Text:\n{chunk_text}\n\nSummary:"
    )
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=256,
            reasoning_effort="low",
            stream=False,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summarization failed: {e}")
        return chunk_text # Fallback to full text if failed

def store_documents_in_qdrant(text_files):
    """
    Chunks documents, embeds (dense + sparse), and stores them in Qdrant for hybrid search.
    """
    embeddings = get_embeddings_model()
    sparse_embeddings = get_sparse_embeddings_model()
    
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
        
    print(f"Total chunks to process: {len(all_chunks)}")
    print("Generating summaries using remote openai/gpt-oss-120b via Groq...")
    
    import time
    
    summarizer_llm = get_summarizer_llm()
    for i, chunk in enumerate(all_chunks):
        if i > 0 and i % 5 == 0:
            print(f"Summarizing chunk {i+1}/{len(all_chunks)}...")
            time.sleep(1)
            
        original_text = chunk.page_content
        summary = summarize_chunk(original_text, summarizer_llm)
        
        # Strategy: Embed the summary, store the raw text safely in metadata
        chunk.metadata["full_content"] = original_text
        chunk.page_content = summary
    
    if all_chunks:
        print("Storing chunks with hybrid (dense + sparse) vectors in Qdrant...")
        if QDRANT_URL and QDRANT_API_KEY:
            vector_store = QdrantVectorStore.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                collection_name=COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                force_recreate=False  # Needed to add sparse vector config to collection
            )
        else:
            vector_store = QdrantVectorStore.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                collection_name=COLLECTION_NAME,
                path="data/qdrant_local",
                force_recreate=False  # Needed to add sparse vector config to collection
            )
        print("Hybrid indexing complete!")
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
