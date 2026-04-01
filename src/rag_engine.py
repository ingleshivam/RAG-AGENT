import os
import re
import textwrap
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from operator import itemgetter
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.vector_store import get_vector_store
from groq import Groq

load_dotenv()
client = Groq()

from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class GroqLLM(LLM):
    model_name: str = "openai/gpt-oss-120b"
    temperature: float = 1.0
    
    @property
    def _llm_type(self) -> str:
        return "custom_groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from groq import Groq
        client = Groq()
        print("🚨 PROMPT : ", prompt)
        args = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_completion_tokens": 2048,
            "top_p": 1,
            "reasoning_effort": "medium",
            "stream": False,
        }
        
        if stop:
            args["stop"] = stop
            
        completion = client.chat.completions.create(**args)
        return completion.choices[0].message.content

def get_llm():
    return GroqLLM(model_name="openai/gpt-oss-120b", temperature=1.0)

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
    prompt_template = """You are a helpful assistant for document analysis. Use the exact provided context to answer the user's question. 
If you don't know the answer, say that you don't know. 

CRITICAL INSTRUCTION: You MUST append the exact page numbers you used to formulate your answer at the very end of your response, formatted exactly like this:
CITED_PAGES: [4, 5]

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
        
        def swap_content(docs):
            """Swaps the summary back to the full text before sending to LLM"""
            for doc in docs:
                if "full_content" in doc.metadata:
                    doc.page_content = doc.metadata["full_content"]
                    
                # Prepend the page number directly to the text so the LLM can cite it
                page = doc.metadata.get("page_number", "Unknown")
                doc.page_content = f"[Source Page: {page}]\n" + doc.page_content
                
            return docs
            
        processed_retriever = itemgetter("input") | retriever | RunnableLambda(swap_content)
        
        retrieval_chain = create_retrieval_chain(processed_retriever, combine_docs_chain)
        
        try:
            response = retrieval_chain.invoke({"input": query})
            
            # Retrieve answer and full context
            all_source_documents = response.get("context", [])
            answer_text = response.get("answer", "")
            
            # Extract the CITED_PAGES array using regex
            cited_pages = []
            match = re.search(r"CITED_PAGES:\s*\[(.*?)\]", answer_text)
            if match:
                pages_str = match.group(1)
                cited_pages = [p.strip() for p in pages_str.split(",") if p.strip()]

            # Filter the source documents to only those explicitly cited by the LLM
            used_source_documents = []
            for doc in all_source_documents:
                page = str(doc.metadata.get("page_number"))
                if page in cited_pages:
                    if doc not in used_source_documents:
                        used_source_documents.append(doc)
                        
            # Fallback if no specific source was properly cited
            if not used_source_documents:
                used_source_documents = all_source_documents
                
            # Clean up the output so CITED_PAGES doesn't show in the Streamlit UI answer text
            answer_text = re.sub(r"CITED_PAGES:\s*\[.*?\]", "", answer_text).strip()

            # Clean up the injected tag before returning to UI
            for doc in used_source_documents:
                page = doc.metadata.get("page_number", "Unknown")
                tag = f"[Source Page: {page}]\n"
                if doc.page_content.startswith(tag):
                    doc.page_content = doc.page_content[len(tag):]

            print("Source Docs cited:", [doc.metadata.get("page_number") for doc in used_source_documents])
            return {
                "answer": answer_text,
                "source_documents": used_source_documents
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
