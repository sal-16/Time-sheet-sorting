"""
Timesheet Detector
==================
Scans a folder of images/PDFs and classifies each file as TIMESHEET or NOT TIMESHEET.

Run from anywhere — always scans Timesheet_Project:
    python detector.py
    python "C:\\Users\\Saad Ullah Khan\\Desktop\\Timesheet_Project\\detector.py"
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract

    for _path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ):
        if os.path.exists(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break
except ImportError:
    pytesseract = None  # type: ignore

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ALWAYS scan this folder by default — no matter where you run the script from.
TIMESHEET_FOLDER = Path(r"C:\Users\Saad Ullah Khan\Desktop\Timesheet_Project")

# Where results CSV is saved.
PROJECT_DIR = TIMESHEET_FOLDER

SKIP_DIR_NAMES = {
    "node_modules", ".git", "__pycache__", "extensions", "resources",
    "locales", "app", "bin", "lib", "include", "share",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}
ALL_EXTS = IMAGE_EXTS | PDF_EXTS

TIMESHEET_THRESHOLD = 50
LIKELY_THRESHOLD = 35
MAX_SIDE = 1600

TIME_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\.\d{1,2}\s*h(?:rs?|ours?)?\b",
    re.I,
)
DATE_RE = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
HOURS_RE = re.compile(r"\b(?:total|weekly|regular|overtime|guaranteed)?\s*hours?\b", re.I)
FILENAME_TS_RE = re.compile(r"(?:\bwe\b|\btime\s*card\b|\btimesheet\b)", re.I)


@dataclass
class Result:
    path: str
    is_timesheet: bool
    label: str
    score: int
    confidence: int
    notes: str

    def line(self) -> str:
        mark = "YES" if self.is_timesheet else "NO "
        return f"  [{mark}]  {self.score:3d}/100  {self.label:<22}  {Path(self.path).name}"


def load_gray(path: str) -> np.ndarray | None:
    """Load image with PIL (memory-safe) and return grayscale numpy array."""
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            longest = max(w, h)
            if longest > MAX_SIDE:
                scale = MAX_SIDE / longest
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )
            rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None


def pdf_first_page_gray(path: str) -> np.ndarray | None:
    if fitz is None:
        return None
    try:
        doc = fitz.open(path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        longest = max(h, w)
        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray
    except Exception:
        return None


def ocr_extract(gray: np.ndarray) -> tuple[str, int]:
    """Return OCR text and estimated row count from one pass."""
    if pytesseract is None:
        return "", 0

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    best_text = ""
    best_rows = 0

    for img in (gray, thresh):
        try:
            data = pytesseract.image_to_data(
                img, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
            )
        except Exception:
            continue

        words: list[str] = []
        ys: list[int] = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            if len(word) < 2:
                continue
            words.append(word.lower())
            ys.append(data["top"][i] + data["height"][i] // 2)

        text = " ".join(words)
        rows: list[int] = []
        for y in sorted(ys):
            if not rows or all(abs(y - ry) > 12 for ry in rows):
                rows.append(y)

        if len(text) > len(best_text):
            best_text = text
            best_rows = len(rows)

    return best_text, best_rows


def filename_score(path: str) -> tuple[int, str]:
    name = Path(path).stem.lower()
    pts = 0
    notes: list[str] = []

    if FILENAME_TS_RE.search(name):
        pts += 18
        notes.append("filename hint")

    if re.search(r"\bwe\s+\d", name):
        pts += 10
        notes.append("week-ending in name")

    return pts, ", ".join(notes)


def score_document(text: str, row_count: int, path: str) -> tuple[int, str]:
    score = 0
    notes: list[str] = []

    strong = {
        "timesheet": 25, "time sheet": 25, "timecard": 25, "time card": 25,
        "clock in": 20, "clock out": 20, "punch in": 20, "punch out": 20,
        "regular hours": 20, "overtime hours": 20, "total hours": 20,
        "hours worked": 18, "pay period": 18, "week ending": 18,
        "employee signature": 15, "supervisor signature": 15, "traveler signature": 18,
        "manager approval": 15, "manager signature": 15,
        "weekly hours": 20, "guaranteed hours": 15,
    }
    for kw, pts in strong.items():
        if kw in text:
            score += pts
            notes.append(kw)

    weak = {
        "employee": 8, "department": 6, "shift": 8, "break": 6,
        "signature": 8, "hours": 8, "period": 6, "traveler": 10,
        "total": 5, "weekly": 8, "certify": 10,
    }
    weak_hits = sum(1 for kw in weak if kw in text)
    score += min(weak_hits * 4, 24)

    days = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    }
    day_hits = sum(1 for d in days if d in text)
    if day_hits >= 5:
        score += 25
        notes.append(f"{day_hits} weekdays")
    elif day_hits >= 3:
        score += 12

    time_hits = len(TIME_RE.findall(text))
    score += min(time_hits * 4, 28)
    if time_hits:
        notes.append(f"{time_hits} times")

    date_hits = len(DATE_RE.findall(text))
    score += min(date_hits * 3, 18)
    if date_hits:
        notes.append(f"{date_hits} dates")

    if HOURS_RE.search(text):
        score += 15
        notes.append("hours field")

    if row_count >= 15:
        score += 30
        notes.append(f"{row_count} rows")
    elif row_count >= 8:
        score += 22
        notes.append(f"{row_count} rows")
    elif row_count >= 5:
        score += 12
        notes.append(f"{row_count} rows")

    reject = 0
    for kw, pts in {
        "invoice": 30, "pivottable": 40, "pivot table": 40,
        "unit price": 20, "subtotal": 20, "purchase order": 25,
    }.items():
        if kw in text:
            reject += pts
    if reject:
        score = max(0, score - reject)
        notes.append("invoice/excel signal")

    fn_pts, fn_note = filename_score(path)
    score += fn_pts
    if fn_note:
        notes.append(fn_note)

    return min(score, 100), ", ".join(notes[:7]) if notes else "no strong signals"


def classify(score: int) -> tuple[bool, str]:
    if score >= TIMESHEET_THRESHOLD:
        return True, "TIMESHEET"
    if score >= LIKELY_THRESHOLD:
        return False, "LIKELY (review manually)"
    return False, "NOT TIMESHEET"


def analyse_gray(gray: np.ndarray, path: str) -> Result:
    text, rows = ocr_extract(gray)
    score, notes = score_document(text, rows, path)
    is_ts, label = classify(score)
    return Result(path, is_ts, label, score, min(100, score), notes)


def process_file(path: str) -> Result:
    ext = Path(path).suffix.lower()

    if ext in IMAGE_EXTS:
        gray = load_gray(path)
        if gray is None:
            return Result(path, False, "UNREADABLE", 0, 0, "could not read image")
        result = analyse_gray(gray, path)
        del gray
        gc.collect()
        return result

    if ext in PDF_EXTS:
        gray = pdf_first_page_gray(path)
        if gray is None:
            msg = "install pymupdf: python -m pip install pymupdf"
            if fitz is None:
                return Result(path, False, "PDF SKIPPED", 0, 0, msg)
            return Result(path, False, "PDF ERROR", 0, 0, "could not render PDF")
        result = analyse_gray(gray, path)
        del gray
        gc.collect()
        return result

    return Result(path, False, "UNSUPPORTED", 0, 0, ext)


def _is_skipped_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIR_NAMES for part in rel.parts[:-1])


def collect_files(folder: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    files: list[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALL_EXTS:
            continue
        if p.name.startswith("."):
            continue
        if recursive and _is_skipped_path(p, folder):
            continue
        files.append(p)
    return sorted(files)


def default_folder() -> Path:
    if not TIMESHEET_FOLDER.is_dir():
        log.error("Timesheet folder not found: %s", TIMESHEET_FOLDER)
        sys.exit(1)
    return TIMESHEET_FOLDER


def warn_if_suspicious_folder(folder: Path) -> None:
    path_str = str(folder).lower()
    suspicious = ("microsoft vs code", "program files", "appdata\\local\\programs")
    if any(s in path_str for s in suspicious):
        log.error(
            "You are scanning a system/VS Code folder, not your timesheet project.\n"
            "  Scanned : %s\n"
            "  Use     : python \"%s\" --folder \"%s\"",
            folder,
            PROJECT_DIR / "detector.py",
            PROJECT_DIR,
        )
        sys.exit(1)


def scan_folder(folder: str, recursive: bool = False) -> list[Result]:
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        log.error("Folder not found: %s", folder_path)
        sys.exit(1)

    warn_if_suspicious_folder(folder_path)
    files = collect_files(folder_path, recursive=recursive)
    if not files:
        log.warning("No image/PDF files found in %s", folder_path)
        return []

    log.info("\n%s", "=" * 68)
    log.info("  TIMESHEET DETECTOR  —  %d file(s) in %s", len(files), folder_path)
    log.info("%s\n", "=" * 68)

    results: list[Result] = []
    for i, fpath in enumerate(files, 1):
        log.info("[%d/%d]  %s", i, len(files), fpath.name)
        result = process_file(str(fpath))
        log.info("%s", result.line())
        if result.notes:
            log.info("         notes: %s", result.notes)
        log.info("")
        results.append(result)

    return results


def print_summary(results: list[Result]) -> None:
    if not results:
        return

    yes = [r for r in results if r.is_timesheet]
    no = [r for r in results if not r.is_timesheet and r.label not in ("UNREADABLE", "PDF ERROR", "PDF SKIPPED")]

    log.info("%s", "=" * 68)
    log.info("  SUMMARY")
    log.info("  Total scanned : %d", len(results))
    log.info("  Timesheets    : %d", len(yes))
    log.info("  Not timesheet : %d", len(no))
    log.info("%s", "=" * 68)

    if yes:
        log.info("\n  TIMESHEETS:")
        for r in yes:
            log.info("    • %s  (%d%%)", Path(r.path).name, r.confidence)

    review = [r for r in results if r.label == "LIKELY (review manually)"]
    if review:
        log.info("\n  REVIEW MANUALLY:")
        for r in review:
            log.info("    • %s  (%d%%)", Path(r.path).name, r.confidence)

    log.info("")


def save_csv(results: list[Result], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "is_timesheet", "label", "score", "confidence", "notes", "full_path"])
        for r in results:
            writer.writerow([
                Path(r.path).name,
                "YES" if r.is_timesheet else "NO",
                r.label,
                r.score,
                r.confidence,
                r.notes,
                r.path,
            ])
    log.info("Results saved to: %s\n", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect timesheet images/PDFs in a folder")
    parser.add_argument(
        "--folder",
        default="",
        help=f"Folder to scan (default: {PROJECT_DIR})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan files inside subfolders",
    )
    parser.add_argument(
        "--out",
        default="",
        help=f"CSV output path (default: {PROJECT_DIR / 'timesheet_results.csv'})",
    )
    args = parser.parse_args()

    if pytesseract is None:
        log.error("pytesseract is not installed. Run: python -m pip install pytesseract")
        sys.exit(1)

    folder = Path(args.folder).resolve() if args.folder else default_folder()
    log.info("\n*** Scanning ONLY this folder: %s ***", folder)
    log.info("(Script location and terminal cwd are IGNORED)\n")

    results = scan_folder(str(folder), recursive=args.recursive)
    print_summary(results)

    out = args.out or str(PROJECT_DIR / "timesheet_results.csv")
    save_csv(results, out)


if __name__ == "__main__":
    main()
