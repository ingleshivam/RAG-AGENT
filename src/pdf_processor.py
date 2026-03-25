import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv

load_dotenv()

# Configure Tesseract path if set
tesseract_cmd = os.getenv("TESSERACT_CMD_PATH")
if tesseract_cmd and os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

poppler_path = os.getenv("POPPLER_PATH")
if poppler_path and not os.path.exists(poppler_path):
    poppler_path = None # Fall back to default if invalid or missing

def process_pdf(pdf_path: str, output_dir: str):
    """
    Processes a PDF file, extracting text page by page.
    Uses PyMuPDF for text extraction. If a page has minimal text (likely an image/scan),
    it falls back to OCR using pdf2image and pytesseract.
    Returns the path to the extracted text file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    doc_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(doc_name)[0]
    output_file_path = os.path.join(output_dir, f"{base_name}.txt")
    
    doc = fitz.open(pdf_path)
    
    with open(output_file_path, "w", encoding="utf-8") as out_txt:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            
            # If text is too short, we might be looking at an image or scanned page
            if len(text) < 50:
                print(f"[{base_name}] Page {page_num + 1} has low text density. Running OCR...")
                # Convert this specific page to an image
                try:
                    images = convert_from_path(
                        pdf_path, 
                        first_page=page_num + 1, 
                        last_page=page_num + 1,
                        poppler_path=poppler_path
                    )
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        text = ocr_text.strip()
                except Exception as e:
                    print(f"[{base_name}] Error running OCR on page {page_num + 1}: {e}")
                    text = f"[OCR Failed for page {page_num + 1}: {e}]"
            
            # Write page header and content
            out_txt.write(f"--- PAGE {page_num + 1} ---\n")
            out_txt.write(text + "\n\n")
            
    return output_file_path

def process_directory(input_dir: str, output_dir: str):
    """
    Iterates over all PDFs in the input directory and processes them.
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        
    extracted_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            out_file = process_pdf(pdf_path, output_dir)
            extracted_files.append(out_file)
            
    return extracted_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process PDFs to extract text with OCR fallback.")
    parser.add_argument("--input", type=str, default="data/raw_pdfs", help="Directory containing input PDFs")
    parser.add_argument("--output", type=str, default="data/extracted_text", help="Directory to save extracted texts")
    args = parser.parse_args()
    
    process_directory(args.input, args.output)
    print("PDF processing complete.")
