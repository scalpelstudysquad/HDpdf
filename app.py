from flask import Flask, request, send_file, send_from_directory
import fitz  # PyMuPDF
import cv2
import numpy as np
import io

app = Flask(__name__)

# --- THE IMAGE MATH ENGINE ---
def peakhd_enhance(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    # Upscale and Denoise
    height, width = img.shape[:2]
    img_hd = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_LANCZOS4)
    img_denoised = cv2.bilateralFilter(img_hd, d=9, sigmaColor=75, sigmaSpace=75)

    # 3-Band USM
    img_f = img_denoised.astype(np.float32)
    bA = cv2.GaussianBlur(img_f, (0, 0), 1.0)
    bB = cv2.GaussianBlur(img_f, (0, 0), 4.0)
    bC = cv2.GaussianBlur(img_f, (0, 0), 14.0)
    
    SA = 1.8
    enhanced_f = img_f + (SA*1.0 * (img_f - bA)) + (SA*0.80 * (bA - bB)) + (SA*0.38 * (bB - bC))
    enhanced_img = np.clip(enhanced_f, 0, 255).astype(np.uint8)

    success, encoded_img = cv2.imencode('.png', enhanced_img)
    return encoded_img.tobytes() if success else None


# --- THE WEB ROUTES ---
@app.route('/')
def home():
    # This serves your HTML website
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return "No file uploaded", 400
        
    file = request.files['file']
    pdf_bytes = file.read()
    
    # Open the uploaded PDF in memory
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    upgraded_xrefs = set()
    
    # Process images safely using PyMuPDF native replace_image
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in upgraded_xrefs: 
                continue
                
            base_image = doc.extract_image(xref)
            new_image_bytes = peakhd_enhance(base_image["image"])
            
            if new_image_bytes:
                page.replace_image(xref, stream=new_image_bytes)
                upgraded_xrefs.add(xref)

    # Save the new PDF to memory and send it back to the browser
    output_pdf = io.BytesIO()
    doc.save(output_pdf, garbage=4, deflate=True)
    output_pdf.seek(0)
    
    return send_file(
        output_pdf, 
        as_attachment=True, 
        download_name="PeakHD_Pro.pdf",
        mimetype="application/pdf"
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

