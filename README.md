# 🕒 Timesheet Detector (AI + OCR Based)

An intelligent document classification system that automatically detects whether a file is a **TIMESHEET** or **NOT TIMESHEET** using OCR, image processing, and rule-based scoring.

---

## 🚀 Features

- 📄 Supports **images + PDFs**
- 🔍 OCR-based text extraction using **Tesseract**
- 🔄 Automatic **rotation correction (0°, 90°, 180°, 270°)**
- 🧠 Hybrid scoring engine (rule-based AI logic)
- 📊 Detects timesheets using:
  - Clock in/out patterns
  - Hours worked fields
  - Pay period detection
  - Column headers
  - Filename hints
- ❌ Strong rejection of:
  - Invoices
  - Project plans
  - Reports
  - Excel/sample data sheets
- 📦 Batch folder processing
- 📑 CSV export of results

---

## ⚙️ How It Works

The system follows this pipeline:

1. **Load File**
   - Reads image or PDF

2. **Preprocessing**
   - Grayscale conversion
   - Resizing large images
   - Noise reduction

3. **Rotation Correction**
   - Detects orientation using Tesseract OSD
   - Fixes rotated documents automatically

4. **OCR Extraction**
   - Extracts text using Tesseract OCR

5. **Scoring Engine**
   - Adds points for timesheet keywords
   - Removes points for non-timesheet signals
   - Uses structural + contextual analysis

6. **Classification**
   - `TIMESHEET`
   - `LIKELY (review manually)`
   - `NOT TIMESHEET`

---

## 📊 Performance

Tested on **55 documents**:

- ✅ 39 correctly identified as timesheets  
- ❌ 16 classified as non-timesheets (mostly low-quality or distant screen captures)

---

## 📁 Project Structure
Timesheet_Project/
│
├── detector.py # Main detection engine
├── main.py # Testing / entry scripts
├── test scripts # Experimentation files
├── requirements.txt # Dependencies
└── README.md


---

## 🛠️ Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Install Tesseract OCR
Windows: Install from
https://github.com/tesseract-ocr/tesseract

Then update path in code if needed:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
▶️ Usage
Run on default folder:
python detector.py
Run on custom folder:
python detector.py --folder "C:\path\to\files"
Save results to CSV:
python detector.py --out results.csv
Enable recursive scanning:
python detector.py --recursive
