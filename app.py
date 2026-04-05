import os
import glob
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from src.pdf_processor import process_directory
from src.vector_store import store_documents_in_qdrant
from src.rag_engine import setup_rag_chain

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

DATA_DIR = "data"
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
EXTRACTED_TEXT_DIR = os.path.join(DATA_DIR, "extracted_text")

os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

# ── Global app state ─────────────────────────────────────────────────────────
rag_chain = None
engine_status = {"state": "initializing", "message": "Starting RAG engine…"}
processing_status = {"state": "idle", "message": ""}


def init_rag_engine():
    global rag_chain, engine_status
    try:
        rag_chain = setup_rag_chain()
        engine_status = {"state": "ready", "message": "RAG engine is ready."}
    except Exception as e:
        engine_status = {"state": "error", "message": str(e)}


def process_and_embed(file_paths):
    global processing_status
    try:
        processing_status = {"state": "processing", "message": "Extracting text & running OCR…"}
        extracted = process_directory(RAW_PDF_DIR, EXTRACTED_TEXT_DIR)

        processing_status = {"state": "embedding", "message": f"Embedding {len(extracted)} file(s) into Qdrant…"}
        store_documents_in_qdrant(extracted)

        processing_status = {"state": "done", "message": f"Successfully processed {len(extracted)} file(s)."}
    except Exception as e:
        processing_status = {"state": "error", "message": str(e)}


# Initialise RAG engine in background on startup
threading.Thread(target=init_rag_engine, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "engine": engine_status,
        "processing": processing_status,
    })


@app.route("/api/documents")
def api_documents():
    txts = glob.glob(os.path.join(EXTRACTED_TEXT_DIR, "*.txt"))
    docs = [os.path.basename(t).replace(".txt", ".pdf") for t in txts]
    return jsonify({"documents": sorted(docs)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global processing_status

    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400

    files = request.files.getlist("files")
    saved = []
    for f in files:
        if f.filename and f.filename.lower().endswith(".pdf"):
            name = secure_filename(f.filename)
            f.save(os.path.join(RAW_PDF_DIR, name))
            saved.append(name)

    if not saved:
        return jsonify({"error": "No valid PDF files received."}), 400

    processing_status = {"state": "queued", "message": f"Queued {len(saved)} file(s) for processing…"}
    threading.Thread(target=process_and_embed, args=(saved,), daemon=True).start()

    return jsonify({"message": f"Uploaded {len(saved)} file(s). Processing started.", "files": saved})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if engine_status["state"] != "ready":
        return jsonify({"error": "RAG engine is not ready yet. Please wait."}), 503

    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    document = data.get("document", "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400
    if not document:
        return jsonify({"error": "Document selection is required."}), 400

    try:
        response = rag_chain(query, document)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"RAG engine error: {str(e)}"}), 500

    answer = response.get("answer", "No answer found.")
    raw_sources = response.get("source_documents", [])

    sources = []
    seen = set()
    for doc in raw_sources:
        page = doc.metadata.get("page_number", "N/A")
        content = doc.page_content
        key = (page, content[:80])
        if key not in seen:
            seen.add(key)
            sources.append({"page": page, "content": content})

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
