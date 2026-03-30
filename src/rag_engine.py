import os
import textwrap
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.vector_store import get_vector_store

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model="gpt-oss:latest",
        temperature=0.1
    )

def setup_rag_chain():
    """
    Sets up the RAG chain using LangChain LCEL.
    Returns a function that can query the chain for a specific document.
    """
    try:
        llm = get_llm()
        vector_store = get_vector_store()
    except Exception as e:
        raise RuntimeError(f"Initialization error in RAG components: {e}")
    
    # Define a prompt template that encourages citation
    prompt_template = """You are a helpful assistant for document analysis. Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer, just say that you don't know. Provide the answer and cite the source pages using the metadata provided.

Context:
{context}

Question: {input}
Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Document combining chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    def query_document(query: str, document_name: str):
        """
        Retrieves context filtered by document_name and generates an answer.
        """
        # Create a retriever specifically filtered by document_name
        # Qdrant uses payload filters. Langchain Qdrant passes filter natively to Qdrant Client.
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_name",
                    match=MatchValue(value=document_name)
                )
            ]
        )
        
        # Langchain Qdrant support allows passing filter in search_kwargs
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": filter_condition
            }
        )
        
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        try:
            response = retrieval_chain.invoke({"input": query})
            
            # Format the context for the UI so it can display citations
            source_documents = response.get("context", [])
            
            return {
                "answer": response.get("answer"),
                "source_documents": source_documents
            }
        except Exception as e:
             return {
                "answer": f"Error running RAG generation: {e}",
                "source_documents": []
            }

    return query_document

if __name__ == "__main__":
    # Simple test shell
    query_fn = setup_rag_chain()
    print("RAG System Initialized.")
    test_q = input("Ask a question: ")
    test_doc = input("Document name (e.g. sample.pdf): ")
    res = query_fn(test_q, test_doc)
    print("Answer:", res["answer"])
    print("Sources:", len(res["source_documents"]))
