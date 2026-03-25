import os
import glob
import streamlit as st
from src.pdf_processor import process_directory
from src.vector_store import store_documents_in_qdrant
from src.rag_engine import setup_rag_chain

st.set_page_config(page_title="RAG PDF Chat", layout="wide")

st.title("Document Analysis RAG System")

DATA_DIR = "data"
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
EXTRACTED_TEXT_DIR = os.path.join(DATA_DIR, "extracted_text")

os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

# Main Application State
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = setup_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}. Is Ollama & Qdrant running?")
        st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

# Sidebar for Upload and Processing
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process & Embed Uploaded Files"):
        if uploaded_files:
            with st.spinner("Saving files locally..."):
                for uploaded_file in uploaded_files:
                    with open(os.path.join(RAW_PDF_DIR, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
            with st.spinner("Extracting text and running OCR if necessary..."):
                extracted_files = process_directory(RAW_PDF_DIR, EXTRACTED_TEXT_DIR)
                st.success(f"Processed {len(extracted_files)} files.")
                
            with st.spinner("Chunking and Embedding into Qdrant..."):
                store_documents_in_qdrant(extracted_files)
                st.success("Documents Embedded successfully!")
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    
    st.subheader("Select Document Space")
    # List all processed documents based on text files
    processed_txts = glob.glob(os.path.join(EXTRACTED_TEXT_DIR, "*.txt"))
    doc_names = [os.path.basename(txt).replace(".txt", ".pdf") for txt in processed_txts]
    
    if doc_names:
        selected_doc = st.selectbox("Choose a document to query", ["None"] + doc_names)
        if selected_doc != "None" and selected_doc != st.session_state.active_document:
            st.session_state.active_document = selected_doc
            st.session_state.messages = []  # Clear chat history when switching docs
            st.success(f"Context switched to {selected_doc}")
    else:
        st.info("No documents processed yet. Upload and Process a document.")

# Main Chat Interface
if st.session_state.active_document:
    st.subheader(f"Chatting with: {st.session_state.active_document}")
    
    # Display chat messages history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(msg["sources"]):
                        page = source.metadata.get('page_number', 'N/A')
                        st.markdown(f"**Page {page}**: \n`{source.page_content[:300]}...`")
            
    # Input box
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response
        with st.chat_message("assistant"):
            if st.session_state.rag_chain:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain(prompt, st.session_state.active_document)
                    answer = response.get("answer", "No answer found.")
                    sources = response.get("source_documents", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources):
                                page = source.metadata.get('page_number', 'N/A')
                                st.markdown(f"**Page {page}**: \n`{source.page_content[:300]}...`")
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
            else:
                st.error("RAG Engine not initialized.")
else:
    st.info("Please select a document from the sidebar to start chatting.")
