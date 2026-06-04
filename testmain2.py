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
    pytesseract = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Change this to your folder path ──────────────────────────────────────────
TIMESHEET_FOLDER = Path(r"C:\Users\Saad Ullah Khan\Desktop\Timesheet_Project")
PROJECT_DIR      = TIMESHEET_FOLDER
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DIR_NAMES = {
    "node_modules", ".git", "__pycache__", "extensions", "resources",
    "locales", "app", "bin", "lib", "include", "share",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS   = {".pdf"}
ALL_EXTS   = IMAGE_EXTS | PDF_EXTS

TIMESHEET_THRESHOLD = 50
LIKELY_THRESHOLD    = 35
MAX_SIDE            = 1600

TIME_RE    = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\.\d{1,2}\s*h(?:rs?|ours?)?\b", re.I)
DATE_RE    = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
HOURS_RE   = re.compile(r"\b(?:total|weekly|regular|overtime|guaranteed)?\s*hours?\b", re.I)
PERCENT_RE = re.compile(r"\b\d{1,3}\s*%")   # progress percentages → NOT a timesheet signal
FILENAME_TS_RE = re.compile(r"(?:\bwe\b|\btime\s*card\b|\btimesheet\b)", re.I)

# ═════════════════════════════════════════════════════════════════════════════
#  KEYWORD TABLES
# ═════════════════════════════════════════════════════════════════════════════

# Strong signals that this IS a timesheet
TIMESHEET_STRONG = {
    "timesheet":            25,
    "time sheet":           25,
    "timecard":             25,
    "time card":            25,
    "clock in":             22,
    "clock out":            22,
    "punch in":             22,
    "punch out":            22,
    "regular hours":        22,
    "overtime hours":       22,
    "total hours":          20,
    "hours worked":         18,
    "pay period":           20,
    "week ending":          20,
    "weekly hours":         20,
    "guaranteed hours":     15,
    "employee signature":   15,
    "supervisor signature": 15,
    "traveler signature":   18,
    "manager approval":     15,
    "manager signature":    15,
    "certify":              12,
}

# Columns that ONLY appear in timesheets (very discriminative)
TIMESHEET_COLUMN_HEADERS = {
    "time in":    20,
    "time out":   20,
    "clock in":   20,
    "clock out":  20,
    "start time": 15,
    "end time":   15,
    "in time":    15,
    "out time":   15,
    "lunch":      10,
    "break":       8,
    "reg hrs":    15,
    "ot hrs":     15,
    "reg":        5,
    "ot":         5,
}

# ── REJECTION signals — these strongly suggest NOT a timesheet ────────────────
#
# The key insight: a project management / generic Excel sheet has columns
# like "Task Name", "Assigned to", "Days Required", "Progress", "Start Date",
# "End Date" etc.  These are NEVER in a timesheet.
#
HARD_REJECT_PHRASES = {
    # Project / task management
    "project name":      40,
    "task name":         40,
    "assigned to":       35,
    "days required":     40,
    "end date":          25,   # timesheets don't have end dates, they have clock-out times
    "progress":          20,
    "% complete":        30,
    "completion":        25,
    "milestone":         30,
    "deliverable":       30,
    "sprint":            35,
    "backlog":           35,
    "story points":      40,
    # Finance / inventory
    "invoice":           35,
    "unit price":        35,
    "subtotal":          30,
    "purchase order":    35,
    "amount due":        35,
    "quantity":          25,
    "sku":               30,
    "product name":      25,
    "item description":  30,
    "vendor":            25,
    # Analytics / reporting
    "pivot table":       40,
    "pivottable":        40,
    "grand total":       20,
    "revenue":           25,
    "profit":            25,
    "expense":           20,
    "budget":            20,
    # HR / other (not timesheet-specific)
    "performance":       20,
    "kpi":               30,
    "score":             15,
    "rating":            15,
    "feedback":          20,
}

# Softer rejection signals (smaller penalty, can be overridden by strong TS hits)
SOFT_REJECT_PHRASES = {
    "start date":        12,   # timesheets have "date", not "start date"
    "due date":          15,
    "sample data":       30,   # title like "Excel Sample Data" is a dead giveaway
    "sample":            15,
    "excel":             20,   # "Excel Sample Data" in title
    "project":           10,
    "task":              10,
    "status":            10,
    "priority":          12,
    "description":       10,
}


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    path:         str
    is_timesheet: bool
    label:        str
    score:        int
    confidence:   int
    notes:        str

    def line(self) -> str:
        mark = "YES" if self.is_timesheet else "NO "
        return f"  [{mark}]  {self.score:3d}/100  {self.label:<22}  {Path(self.path).name}"


def load_gray(path: str) -> np.ndarray | None:
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
        doc  = fitz.open(path)
        page = doc.load_page(0)
        pix  = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
        rgb  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
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


# ═════════════════════════════════════════════════════════════════════════════
#  OCR
# ═════════════════════════════════════════════════════════════════════════════

def ocr_extract(gray: np.ndarray) -> tuple[str, int]:
    if pytesseract is None:
        return "", 0

    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
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
        ys:    list[int] = []
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


# ═════════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE  (the heart of the fix)
# ═════════════════════════════════════════════════════════════════════════════

def filename_score(path: str) -> tuple[int, str]:
    name = Path(path).stem.lower()
    pts  = 0
    notes: list[str] = []

    if FILENAME_TS_RE.search(name):
        pts += 18
        notes.append("filename hint")
    if re.search(r"\bwe\s+\d", name):
        pts += 10
        notes.append("week-ending in name")

    # Reject filenames that clearly aren't timesheets
    if re.search(r"\b(invoice|project|budget|report|sample|excel)\b", name):
        pts -= 20
        notes.append("non-timesheet filename")

    return pts, ", ".join(notes)


def score_document(text: str, row_count: int, path: str) -> tuple[int, str]:
    """
    Three-pass scoring:
      PASS 1 – Add points for timesheet signals
      PASS 2 – Subtract hard rejection penalties (project/invoice/Excel keywords)
      PASS 3 – Apply soft rejections and contextual adjustments

    Returns (final_score 0-100, human-readable notes).
    """
    score = 0
    notes: list[str] = []

    # ── PASS 1: Timesheet positive signals ───────────────────────────────────

    for kw, pts in TIMESHEET_STRONG.items():
        if kw in text:
            score += pts
            notes.append(kw)

    for kw, pts in TIMESHEET_COLUMN_HEADERS.items():
        if kw in text:
            score += pts
            notes.append(f"col:{kw}")

    # Day-of-week cluster (strong: all 5 weekdays in one doc = it's a weekly timesheet)
    days_full  = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    days_short = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    day_hits   = sum(1 for d in days_full  if d in text)
    day_hits  += sum(1 for d in days_short if d in text and len(d) == 3)
    if day_hits >= 5:
        score += 25
        notes.append(f"{day_hits} weekdays")
    elif day_hits >= 3:
        score += 12

    # Time values  (HH:MM or decimal hours — the core of a timesheet)
    time_hits = len(TIME_RE.findall(text))
    score    += min(time_hits * 4, 28)
    if time_hits:
        notes.append(f"{time_hits} time-values")

    # Date values (moderate signal — also in invoices, so capped low here)
    date_hits = len(DATE_RE.findall(text))
    score    += min(date_hits * 2, 10)   # was 3×18 = too generous

    # "hours" keyword
    if HOURS_RE.search(text):
        score += 12
        notes.append("hours field")

    # Row count — but only if rejection score is low (added after rejection check below)
    raw_row_bonus = 0
    if row_count >= 15:
        raw_row_bonus = 20
    elif row_count >= 8:
        raw_row_bonus = 14
    elif row_count >= 5:
        raw_row_bonus = 8

    # Weak signals (employee, dept, etc.) — small bonus
    weak = {"employee": 6, "department": 5, "shift": 7, "signature": 7,
            "period": 5, "traveler": 9, "weekly": 7, "certify": 9}
    weak_bonus = sum(pts for kw, pts in weak.items() if kw in text)
    score += min(weak_bonus, 20)

    # ── PASS 2: Hard rejection (project/invoice/Excel keywords) ──────────────

    reject = 0
    reject_reasons: list[str] = []

    for kw, pts in HARD_REJECT_PHRASES.items():
        if kw in text:
            reject += pts
            reject_reasons.append(f"-{pts}:{kw}")

    # Progress percentages are a strong "NOT timesheet" signal
    # (timesheets have hours as decimals, not % complete values)
    pct_hits = len(PERCENT_RE.findall(text))
    if pct_hits >= 3:
        reject += 20
        reject_reasons.append(f"-20:progress%x{pct_hits}")
    elif pct_hits >= 1:
        reject += 8

    # "Start date" + "End date" together = project table, not timesheet
    if "start date" in text and "end date" in text:
        reject += 25
        reject_reasons.append("-25:start+end_date_pair")

    # Many different person names as data (assigned-to column) vs. a single employee name
    # Heuristic: if text has 5+ single capitalised words that look like first names
    # and we also have project/task hits, boost rejection
    if reject >= 30:
        reject_reasons.append("(hard reject active)")
        notes.extend(reject_reasons[:5])

    # Apply rejection — but preserve strong timesheet hits
    # If we have overwhelming timesheet signals (score > 60) AND low rejection, keep it
    if reject > 0:
        if score > 60 and reject < 30:
            # Genuine timesheet with some incidental words — reduce rejection impact
            score = max(0, score - reject // 2)
        else:
            score = max(0, score - reject)

    # ── PASS 3: Soft rejections & contextual adjustments ─────────────────────

    soft_reject = 0
    for kw, pts in SOFT_REJECT_PHRASES.items():
        if kw in text:
            soft_reject += pts

    # Only apply soft rejection if we didn't already confirm it as a timesheet
    if score < 60:
        score = max(0, score - soft_reject // 2)

    # Now add row bonus — but only if score still looks like a timesheet
    # (prevents a project table with many rows from scoring higher)
    if score >= 20:
        score += raw_row_bonus

    # ── PASS 3b: Column-header pattern analysis ───────────────────────────────
    # A timesheet's first row is: day | date | in | out | break | hours | total
    # A project sheet's first row is: project | task | assigned | start | end | progress
    # Count how many of each pattern's words appear
    ts_header_words  = {"in", "out", "break", "lunch", "regular", "overtime", "reg", "ot",
                        "time in", "time out", "clock in", "clock out"}
    prj_header_words = {"task", "project", "assigned", "progress", "status", "priority",
                        "milestone", "sprint", "start date", "end date", "due", "days required"}

    ts_header_hits  = sum(1 for w in ts_header_words  if w in text)
    prj_header_hits = sum(1 for w in prj_header_words if w in text)

    if prj_header_hits >= 3 and prj_header_hits > ts_header_hits:
        penalty = (prj_header_hits - ts_header_hits) * 8
        score   = max(0, score - penalty)
        notes.append(f"prj_header_hits={prj_header_hits}")

    # ── filename signal ───────────────────────────────────────────────────────
    fn_pts, fn_note = filename_score(path)
    score += fn_pts
    if fn_note:
        notes.append(fn_note)

    final = min(score, 100)
    return final, ", ".join(notes[:8]) if notes else "no strong signals"


# ═════════════════════════════════════════════════════════════════════════════
#  CLASSIFY
# ═════════════════════════════════════════════════════════════════════════════

def classify(score: int) -> tuple[bool, str]:
    if score >= TIMESHEET_THRESHOLD:
        return True,  "TIMESHEET"
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
        del gray; gc.collect()
        return result

    if ext in PDF_EXTS:
        gray = pdf_first_page_gray(path)
        if gray is None:
            msg = "install pymupdf: python -m pip install pymupdf"
            if fitz is None:
                return Result(path, False, "PDF SKIPPED", 0, 0, msg)
            return Result(path, False, "PDF ERROR", 0, 0, "could not render PDF")
        result = analyse_gray(gray, path)
        del gray; gc.collect()
        return result

    return Result(path, False, "UNSUPPORTED", 0, 0, ext)


# ═════════════════════════════════════════════════════════════════════════════
#  FOLDER SCANNING
# ═════════════════════════════════════════════════════════════════════════════

def _is_skipped_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIR_NAMES for part in rel.parts[:-1])


def collect_files(folder: Path, recursive: bool = False) -> list[Path]:
    candidates = folder.rglob("*") if recursive else folder.iterdir()
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
            "  Scanned : %s\n  Use     : python \"%s\" --folder \"%s\"",
            folder, PROJECT_DIR / "detector.py", PROJECT_DIR,
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


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY & CSV
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[Result]) -> None:
    if not results:
        return

    yes    = [r for r in results if r.is_timesheet]
    no     = [r for r in results if not r.is_timesheet
              and r.label not in ("UNREADABLE", "PDF ERROR", "PDF SKIPPED")]
    review = [r for r in results if r.label == "LIKELY (review manually)"]

    log.info("%s", "=" * 68)
    log.info("  SUMMARY")
    log.info("  Total scanned : %d", len(results))
    log.info("  Timesheets    : %d", len(yes))
    log.info("  Not timesheet : %d", len(no))
    log.info("%s", "=" * 68)

    if yes:
        log.info("\n  ✅ TIMESHEETS:")
        for r in yes:
            log.info("    • %s  (%d%%)", Path(r.path).name, r.confidence)
    if review:
        log.info("\n  🟡 REVIEW MANUALLY:")
        for r in review:
            log.info("    • %s  (%d%%)", Path(r.path).name, r.confidence)
    if not yes and not review:
        log.info("\n  ❌ No timesheets found.")
    log.info("")


def save_csv(results: list[Result], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "is_timesheet", "label", "score", "confidence", "notes", "full_path"])
        for r in results:
            writer.writerow([
                Path(r.path).name,
                "YES" if r.is_timesheet else "NO",
                r.label, r.score, r.confidence, r.notes, r.path,
            ])
    log.info("Results saved to: %s\n", out_path)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect timesheet images/PDFs in a folder")
    parser.add_argument("--folder",    default="",  help=f"Folder to scan (default: {PROJECT_DIR})")
    parser.add_argument("--recursive", action="store_true", help="Also scan subfolders")
    parser.add_argument("--out",       default="",  help="CSV output path")
    args = parser.parse_args()

    if pytesseract is None:
        log.error("pytesseract is not installed. Run: python -m pip install pytesseract")
        sys.exit(1)

    folder = Path(args.folder).resolve() if args.folder else default_folder()
    log.info("\n*** Scanning: %s ***\n", folder)

    results = scan_folder(str(folder), recursive=args.recursive)
    print_summary(results)

    out = args.out or str(PROJECT_DIR / "timesheet_results.csv")
    save_csv(results, out)


if __name__ == "__main__":
    main()