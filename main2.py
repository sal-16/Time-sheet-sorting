import cv2
import numpy as np
import os
import zipfile
import pytesseract
import re

# =========================
# SET YOUR TESSERACT PATH
# =========================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# =========================
# OCR FUNCTION
# =========================
def get_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    return text.lower()


# =========================
# SCORING FUNCTION
# =========================
def score_text(text):
    score = 0

    keywords = [
        "timesheet", "timecard", "employee", "hours",
        "clock in", "clock out", "total hours",
        "break", "signature", "date", "day"
    ]

    # keyword scoring
    for k in keywords:
        if k in text:
            score += 10

    # time pattern (08:30 etc)
    time_matches = re.findall(r'\b\d{1,2}:\d{2}\b', text)
    score += min(len(time_matches) * 5, 30)

    # date pattern (10/05/2025 etc)
    date_matches = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
    score += min(len(date_matches) * 5, 30)

    return score


# =========================
# CLASSIFICATION
# =========================
def classify(score):
    if score >= 70:
        return "YES (Timesheet)"
    elif score >= 40:
        return "LIKELY Timesheet"
    else:
        return "NO (Not Timesheet)"


# =========================
# PROCESS IMAGE
# =========================
def process_image(img):
    text = get_text(img)
    score = score_text(text)
    result = classify(score)
    confidence = min(100, int(score / 100 * 100))

    return result, confidence, score


# =========================
# AUTO FIND ZIP FILE
# =========================
def find_zip():
    for f in os.listdir():
        if f.endswith(".zip"):
            return f
    return None


# =========================
# PROCESS FILE
# =========================
def handle_file(file_path):
    img = cv2.imread(file_path)

    if img is None:
        return "SKIPPED (Unreadable file)", 0, 0

    return process_image(img)


# =========================
# ZIP SCANNER
# =========================
def scan_zip(zip_path):
    print("\n==============================")
    print(" TIMESHEET DETECTOR ")
    print("==============================\n")

    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()

        count = 0

        for f in files:
            if f.startswith("__MACOSX"):
                continue

            if f.lower().endswith((".png", ".jpg", ".jpeg")):

                count += 1
                extracted = z.extract(f)

                result, conf, score = handle_file(extracted)

                print(f"[{count}] {f}")
                print(f"    Result: {result}")
                print(f"    Confidence: {conf}%")
                print(f"    Score: {score}\n")

        if count == 0:
            print("No images found in ZIP")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    zip_file = find_zip()

    if not zip_file:
        print("ERROR: No ZIP file found in folder")
    else:
        scan_zip(zip_file)