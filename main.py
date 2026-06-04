import cv2
import numpy as np
import os
import re
import pytesseract

# Link Tesseract executable directly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_clean_words_and_boxes(image_path):
    """Loads an image, handles errors, and returns filtered low-case text chunks."""
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Scale up small images to protect text features from blurring
    h_img, w_img = gray.shape
    if w_img < 1600:
        scale = 2
        gray = cv2.resize(gray, (w_img * scale, h_img * scale), interpolation=cv2.INTER_CUBIC)

    # Adaptive binarization to clean up lighting gradients and noise shadows
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)

    # Sparse text tracking layout analysis
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
    
    parsed_words = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip().lower()
        if len(text) > 1: # Discard single letter/number artifacts
            parsed_words.append({
                'text': text,
                'center_y': data['top'][i] + (data['height'][i] // 2)
            })
    return parsed_words

def process_single_document(image_path):
    """Checks an image and determines a timesheet match using a weighted score."""
    words = extract_clean_words_and_boxes(image_path)
    if not words:
        return "NO (Unreadable / No Text Found)"

    # --- WORDS MATCHING GROUPS ---
    timesheet_anchors = {"hours", "timesheet", "time-sheet", "overtime", "rate", "employee", "total", "signature", "clock", "name", "period"}
    calendar_anchors = {"mon", "tue", "wed", "thu", "fri", "sat", "sun", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "date", "week", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
    excel_ui_keywords = {"pivottable", "pivot", "fields", "defer layout", "choose fields"}

    anchor_hits = 0
    calendar_hits = 0
    excel_ui_hits = 0

    # Match time formats: standard (13:45, 08.00) or decimal values (8.0, 7.5, 40.0)
    time_regex = re.compile(r'(\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b)|(\b[0-9]\.[0-5]\b)|(\b[1-4][0-9]\.[0-5]\b)')
    time_format_hits = 0

    for word in words:
        txt = word['text']
        if txt in timesheet_anchors:
            anchor_hits += 1
        if any(day in txt for day in calendar_anchors):
            calendar_hits += 1
        if any(ui in txt for ui in excel_ui_keywords):
            excel_ui_hits += 1
        if time_regex.search(txt):
            time_format_hits += 1

    # Absolute Security Rejection Gate
    if excel_ui_hits >= 2:
        return "NO (Excel Interface Dump)"

    # --- ROW ALIGNMENT COUNTING ---
    y_coordinates = [word['center_y'] for word in words]
    unique_rows = []
    for y in sorted(y_coordinates):
        if not unique_rows or all(abs(y - existing_y) > 15 for existing_y in unique_rows):
            unique_rows.append(y)
    total_data_rows = len(unique_rows)

    # =========================================================================
    # THE WEIGHTED SCORING SYSTEM
    # =========================================================================
    score = 0
    
    # 1. Structural Rows (Tables have high row concentration)
    if total_data_rows >= 5:
        score += 30
        
    # 2. Tracking Terms Found
    if anchor_hits >= 1:
        score += 25
        
    # 3. Calendar Day/Month Indicators Found
    if calendar_hits >= 1:
        score += 25
        
    # 4. Numerical Time/Hour Entries Found
    if time_format_hits >= 2:
        score += 20

    # Verification threshold: Real timesheets will easily hit 50+ points.
    # Random images of pets, people, or scenery will fail to gain points across categories.
    if score >= 50:
        return f"YES (Verified | Score: {score}/100 | Rows: {total_data_rows}, Keywords: {anchor_hits + calendar_hits})"
    else:
        return f"NO (Non-Timesheet Layout | Score: {score}/100)"

def scan_entire_workspace():
    workspace_dir = os.getcwd()
    print(f"=== Scanning Workspace Directory: {workspace_dir} ===")
    
    valid_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
    all_files = os.listdir(workspace_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("[ALERT] No image files found in the active workspace folder.")
        return

    print(f"Found {len(image_files)} images to process.\n" + "-"*60)
    
    for idx, img_name in enumerate(image_files, 1):
        full_path = os.path.join(workspace_dir, img_name)
        result = process_single_document(full_path)
        print(f"[{idx:02d}] File: {img_name:<25} -> Result: {result}")
        
    print("-"*60 + "\nBatch Scan Processing Finished Successfully.")

if __name__ == "__main__":
    scan_entire_workspace()
