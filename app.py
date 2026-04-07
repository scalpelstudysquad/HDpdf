# --- THE IMAGE MATH ENGINE ---
def peakhd_enhance(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    height, width = img.shape[:2]
    
    # RAM SAFETY VALVE: 
    # If the image is already wider than 1500px, don't upscale it (prevents 512MB crashes)
    # Otherwise, do a 2x upscale instead of 3x to save memory.
    if width > 1500 or height > 1500:
        img_hd = img 
    else:
        img_hd = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        
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

