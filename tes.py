import cv2
import numpy as np
import os
import zipfile
import pytesseract
import re

# -----------------------------
# SET PATH (IMPORTANT)
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# -----------------------------
# OCR FUNCTION
# -----------------------------
def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # improve OCR (VERY IMPORTANT)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    return text.lower()


# -----------------------------
# STRUCTURE DETECTION (KEY FIX)
# -----------------------------
def detect_table_score(img):
    if img is None:
        return 0

    # -----------------------------
    # SAFETY: resize large images
    # -----------------------------
    h, w = img.shape[:2]

    max_width = 1200
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # grayscale (safe now)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # light blur reduces memory spikes
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # NOW edge detection (safe)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None:
        return 0

    horizontal = 0
    vertical = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angle < 10 or angle > 170:
            horizontal += 1
        elif 80 < angle < 100:
            vertical += 1

    score = min((horizontal + vertical) * 2, 100)

    return score

# -----------------------------
# TEXT SCORE (LIGHTWEIGHT)
# -----------------------------
def text_score(text):
    score = 0

    # soft signals (NOT strict)
    soft_keywords = [
        "hours", "employee", "time", "in", "out",
        "date", "day", "total", "break"
    ]

    for k in soft_keywords:
        if k in text:
            score += 8

    # time patterns (VERY IMPORTANT for timesheets)
    time_hits = len(re.findall(r'\b\d{1,2}:\d{2}\b', text))
    score += time_hits * 10

    # date patterns
    date_hits = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
    score += date_hits * 8

    return min(score, 100)


# -----------------------------
# FINAL CLASSIFIER
# -----------------------------
def classify(text_s, struct_s):
    total = (text_s * 0.5) + (struct_s * 0.5)

    if total >= 60:
        return "YES (Timesheet)"
    elif total >= 40:
        return "LIKELY Timesheet"
    else:
        return "NO (Not Timesheet)"


# -----------------------------
# PROCESS IMAGE
# -----------------------------
def process(img):
    text = extract_text(img)

    t_score = text_score(text)
    s_score = detect_table_score(img)

    result = classify(t_score, s_score)

    return result, t_score, s_score


# -----------------------------
# HANDLE FILE
# -----------------------------
def handle_file(path):
    img = cv2.imread(path)

    if img is None:
        return "ERROR", 0, 0

    return process(img)


# -----------------------------
# AUTO ZIP FINDER
# -----------------------------
def scan_zip(zip_file):
    print("\n===== TIMESHEET DETECTOR =====\n")

    with zipfile.ZipFile(zip_file, 'r') as z:

        files = z.namelist()

        # DEBUG: show EVERYTHING inside zip
        print("FILES INSIDE ZIP:\n")
        for f in files:
            print(" -", f)
        print("\n------------------------\n")

        count = 0

        for f in files:

            # skip system junk
            if "__MACOSX" in f:
                continue

            ext = os.path.splitext(f)[1].lower()

            # accept ALL relevant formats
            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:

                count += 1
                extracted = z.extract(f)

                result, t, s = handle_file(extracted)

                print(f"[{count}] {f}")
                print(f"    Result: {result}")
                print(f"    TextScore: {t}")
                print(f"    StructureScore: {s}\n")

        print(f"\nTOTAL IMAGES PROCESSED: {count}")

# -----------------------------
# ZIP SCANNER
# -----------------------------
def scan_zip(zip_file):
    print("\n===== TIMESHEET DETECTOR =====\n")

    with zipfile.ZipFile(zip_file, 'r') as z:
        files = z.namelist()

        count = 0

        for f in files:
            if f.startswith("__MACOSX"):
                continue

            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                count += 1

                extracted = z.extract(f)

                result, t, s = handle_file(extracted)

                print(f"[{count}] {f}")
                print(f"    Result: {result}")
                print(f"    TextScore: {t}")
                print(f"    StructureScore: {s}\n")

        if count == 0:
            print("No images found")


# -----------------------------
# FIND ZIP FILE
# -----------------------------
def find_zip():
    for f in os.listdir('.'):
        if f.lower().endswith('.zip'):
            return f
    return None


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    zip_file = find_zip()

    if not zip_file:
        print("ERROR: No ZIP file found")
    else:
        scan_zip(zip_file)