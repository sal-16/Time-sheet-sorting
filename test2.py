"""
Timesheet Detector
==================
Classifies images / PDFs as TIMESHEET or NOT TIMESHEET.

Handles:
  - Rotated documents (90° / 180° / 270°) — auto-corrected
  - Handwritten timesheets, clipart/digital timesheets
  - Healthcare & staffing agency forms (AMN, HealthCare Support, Kronos, etc.)
  - Military-time values (0700, 1730 …)

Run:
    python detector.py
    python detector.py --folder C:\\other\\path
    python detector.py --out results.csv
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

# ── optional: pytesseract ────────────────────────────────────────────────────
try:
    import pytesseract
    for _p in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ):
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
    _OCR = True
except ImportError:
    pytesseract = None  # type: ignore
    _OCR = False

# ── optional: PyMuPDF for PDFs ───────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _PDF = True
except ImportError:
    fitz = None  # type: ignore
    _PDF = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── CHANGE THIS to your folder ────────────────────────────────────────────────
TIMESHEET_FOLDER = Path(r"C:\Users\Saad Ullah Khan\Desktop\Timesheet_Project")
PROJECT_DIR      = TIMESHEET_FOLDER
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DIRS  = {"node_modules",".git","__pycache__","extensions","resources",
              "locales","app","bin","lib","include","share"}
IMAGE_EXTS = {".png",".jpg",".jpeg",".bmp",".tiff",".tif",".webp"}
PDF_EXTS   = {".pdf"}
ALL_EXTS   = IMAGE_EXTS | PDF_EXTS

TIMESHEET_THRESHOLD = 50
LIKELY_THRESHOLD    = 35
MAX_SIDE            = 1800   # slightly larger for better OCR on dense docs

# ── regex patterns ────────────────────────────────────────────────────────────
TIME_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b"          # 9:00 / 9.00 / 9:00 AM
    r"|\b\d{1,2}\.\d{1,2}\s*h(?:rs?|ours?)?\b"    # 8.5 hrs
    r"|\b(?:[01]\d|2[0-3])[0-5]\d\b",              # military: 0700, 1730
    re.I,
)
DATE_RE    = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
HOURS_RE   = re.compile(r"\b(?:total|weekly|regular|overtime|guaranteed|call)?\s*hours?\b", re.I)
PERCENT_RE = re.compile(r"\b\d{1,3}\s*%")
FILENAME_TS_RE = re.compile(r"(?:\bwe\b|\btime[\s_]?card\b|\btimesheet\b|\bts\b)", re.I)

# ═════════════════════════════════════════════════════════════════════════════
#  KEYWORD TABLES
# ═════════════════════════════════════════════════════════════════════════════

# --- Strong positive signals ------------------------------------------------
TIMESHEET_STRONG = {
    # Document type identifiers
    "timesheet":              28,
    "time sheet":             28,
    "timecard":               28,
    "time card":              28,
    "weekly timesheet":       35,
    "employee timesheet":     35,
    # Clock actions
    "clock in":               22,  "clock out":            22,
    "punch in":               22,  "punch out":            22,
    # Hours labels
    "regular hours":          22,  "overtime hours":       22,
    "total hours":            20,  "hours worked":         20,
    "total weekly hours":     25,  "weekly hours":         20,
    "guaranteed hours":       18,
    # Shift labels
    "shift start":            22,  "shift end":            22,
    # Lunch break — specific to timesheets
    "lunch out":              20,  "lunch in":             20,
    # On-call / call-back (healthcare timesheets)
    "on-call":                14,  "on call":              14,
    "total call":             16,  "call back":            14,
    "total on-call":          20,  "total call back":      18,
    # Pay period
    "pay period":             22,
    "week ending":            22,  "week's end":           22,
    "week end date":          22,  "weeks end":            20,
    # Signatures & approval
    "employee signature":     16,  "supervisor signature": 16,
    "traveler signature":     20,  "manager signature":    16,
    "manager approval":       16,  "manager print":        14,
    "provider signature":     20,  "authorized approver":  20,
    # Certification language
    "i certify":              16,  "certify that":         16,
    "certify":                12,  "affirm":               12,
    # Staffing / agency context
    "contractor":             12,  "contractor's name":    20,
    "client name":            14,  "traveler":             14,
    # Time-tracking systems
    "kronos":                 20,  "exception log":        16,
    "time keeper":            16,  "badge":                10,
    # Healthcare context
    "clinician":              12,  "facility":              8,
    "mileage":                 8,
}

# --- Column-header signals (these column names ONLY appear in timesheets) ---
TIMESHEET_COLUMN_HEADERS = {
    "time in":       22,  "time out":      22,
    "clock in":      22,  "clock out":     22,
    "shift start":   22,  "shift end":     22,
    "lunch out":     20,  "lunch in":      20,
    "start time":    16,  "end time":      16,
    "in time":       16,  "out time":      16,
    "call hrs":      16,  "call back hrs": 16,
    "total hrs":     14,  "reg hrs":       16,
    "ot hrs":        16,  "reg":            5,
    "ot":             5,  "lunch":         10,
    "break":          8,
}

# --- Hard rejection signals — these words strongly mean NOT a timesheet -----
# NOTE: "end date" removed — "Week's End Date" would falsely trigger it.
#       Instead we only penalise "start date" + "end date" TOGETHER (pair check below).
HARD_REJECT_PHRASES = {
    # Project / task management
    "project name":      40,  "task name":         40,
    "assigned to":       35,  "days required":     40,
    "progress":          22,  "% complete":        30,
    "completion":        25,  "milestone":         30,
    "deliverable":       30,  "sprint":            35,
    "backlog":           35,  "story points":      40,
    # Finance / inventory
    "invoice":           35,  "unit price":        35,
    "subtotal":          30,  "purchase order":    35,
    "amount due":        35,  "quantity":          25,
    "sku":               30,  "product name":      25,
    "item description":  30,  "vendor":            25,
    # Analytics / reporting
    "pivot table":       40,  "pivottable":        40,
    "grand total":       22,  "revenue":           25,
    "profit":            25,  "expense":           22,
    "budget":            22,
}

# --- Soft rejection signals (small penalty, overrideable by strong TS hits) --
SOFT_REJECT_PHRASES = {
    "sample data":    30,   # "Excel Sample Data" title
    "excel sample":   30,
    "sample":         12,
    "due date":       14,
    "start date":     10,   # alone it's weak; pair check is the main handler
    "project":         8,
    "task":            8,
    "priority":       10,
    "status":          5,
    "description":     5,
}


# ═════════════════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
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
        bar  = "█" * (self.confidence // 5) + "░" * (20 - self.confidence // 5)
        return (f"  [{mark}]  [{bar}] {self.confidence:3d}%  "
                f"{self.label:<26}  {Path(self.path).name}")


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_gray(path: str) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            longest = max(w, h)
            if longest > MAX_SIDE:
                scale = MAX_SIDE / longest
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    except Exception:
        return None


def pdf_first_page_gray(path: str) -> np.ndarray | None:
    if not _PDF:
        return None
    try:
        doc  = fitz.open(path)
        page = doc.load_page(0)
        pix  = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
        rgb  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  ROTATION CORRECTION  (this is the critical fix)
# ═════════════════════════════════════════════════════════════════════════════

def _rotate(gray: np.ndarray, angle: int) -> np.ndarray:
    """Apply a 0/90/180/270 clockwise rotation."""
    if angle == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    return gray  # 0 degrees — no change


def _word_count(gray: np.ndarray) -> int:
    """Quick OCR on a downscaled image — used only for rotation ranking."""
    if not _OCR:
        return 0
    h, w = gray.shape
    if max(h, w) > 500:
        s = 500 / max(h, w)
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        small = gray
    try:
        txt = pytesseract.image_to_string(small, config="--oem 3 --psm 3")
        return len([t for t in txt.split() if len(t) >= 3])
    except Exception:
        return 0


def correct_rotation(gray: np.ndarray) -> np.ndarray:
    """
    Detect and fix page rotation.

    Strategy:
      1. Ask Tesseract OSD for the rotation angle.  If OSD is confident,
         apply it immediately.
      2. If OSD fails or has low confidence, try all four orientations and
         return whichever one produces the most readable OCR words.
         This handles cases where OSD cannot find enough characters.
    """
    if not _OCR:
        return gray

    # ── Step 1: Tesseract OSD ─────────────────────────────────────────────
    try:
        osd = pytesseract.image_to_osd(
            gray, config="--psm 0 -c min_characters_to_try=5"
        )
        angle_m = re.search(r"Rotate:\s+(\d+)", osd)
        conf_m  = re.search(r"Orientation confidence:\s+([\d.]+)", osd)

        angle = int(angle_m.group(1)) if angle_m else 0
        conf  = float(conf_m.group(1)) if conf_m else 0.0

        if conf >= 1.5:          # OSD is confident → trust it
            corrected = _rotate(gray, angle)
            log.debug("  OSD rotation: %d° (conf=%.1f)", angle, conf)
            return corrected

        # OSD says 0° with moderate confidence — probably fine
        if conf >= 0.8 and angle == 0:
            return gray

        # Low confidence — fall through to brute force
        log.debug("  OSD low-confidence (%.1f) — trying all rotations", conf)

    except Exception:
        log.debug("  OSD failed — trying all rotations")

    # ── Step 2: Brute-force — try all 4 orientations ─────────────────────
    candidates = [(0, gray)]
    for a in (90, 180, 270):
        candidates.append((a, _rotate(gray, a)))

    best_angle, best_img = max(candidates, key=lambda t: _word_count(t[1]))
    if best_angle != 0:
        log.debug("  Brute-force chose rotation: %d°", best_angle)
    return best_img


# ═════════════════════════════════════════════════════════════════════════════
#  OCR EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _preprocess_variants(gray: np.ndarray) -> list[np.ndarray]:
    """Return a list of preprocessed images to try OCR on."""
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold — best for handwritten / low-contrast docs
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    # Otsu — best for clean printed docs
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [gray, thresh, otsu]


def ocr_extract(gray: np.ndarray) -> tuple[str, int]:
    """
    Run OCR with multiple preprocessing variants and PSM modes.
    Returns (best_text, row_count).
    """
    if not _OCR:
        return "", 0

    best_text = ""
    best_rows = 0

    for img in _preprocess_variants(gray):
        for psm in ("6", "4", "3"):   # uniform block, single column, auto
            cfg = f"--oem 3 --psm {psm}"
            try:
                data = pytesseract.image_to_data(
                    img, config=cfg, output_type=pytesseract.Output.DICT
                )
            except Exception:
                continue

            words: list[str] = []
            ys:    list[int] = []
            for i, w in enumerate(data["text"]):
                w = (w or "").strip()
                if len(w) < 2:
                    continue
                conf = int(data.get("conf", [0])[i] or 0)
                if conf < 20:   # skip very low-confidence tokens
                    continue
                words.append(w.lower())
                ys.append(data["top"][i] + data["height"][i] // 2)

            text = " ".join(words)

            # Count distinct text rows
            rows: list[int] = []
            for y in sorted(ys):
                if not rows or all(abs(y - ry) > 12 for ry in rows):
                    rows.append(y)

            if len(text) > len(best_text):
                best_text = text
                best_rows = len(rows)

    return best_text, best_rows


# ═════════════════════════════════════════════════════════════════════════════
#  FILENAME SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

def filename_score(path: str) -> tuple[int, str]:
    name = Path(path).stem.lower()
    pts: int = 0
    notes: list[str] = []

    if FILENAME_TS_RE.search(name):
        pts += 20
        notes.append("filename:timesheet")
    if re.search(r"we[\s_\-]\d|week[\s_\-]end", name):
        pts += 14
        notes.append("filename:week-ending")
    if re.search(r"\b(invoice|project|budget|report|sample|excel|pivot)\b", name):
        pts -= 20
        notes.append("filename:non-timesheet")

    return pts, ", ".join(notes)


# ═════════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def score_document(text: str, row_count: int, path: str) -> tuple[int, str]:
    """
    Three-pass scoring system.

    PASS 1 — Add points for timesheet signals
    PASS 2 — Subtract hard-rejection penalties (project/invoice/Excel)
    PASS 3 — Soft rejections + row bonus + filename + column-header pattern
    """
    score = 0
    notes: list[str] = []

    # ── PASS 1: Positive signals ──────────────────────────────────────────

    for kw, pts in TIMESHEET_STRONG.items():
        if kw in text:
            score += pts
            notes.append(kw)

    for kw, pts in TIMESHEET_COLUMN_HEADERS.items():
        if kw in text:
            score += pts
            notes.append(f"col:{kw}")

    # Days-of-week cluster
    full_days  = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
    short_days = {"mon","tue","wed","thu","fri","sat","sun"}
    day_hits   = sum(1 for d in full_days  if d in text)
    day_hits  += sum(1 for d in short_days if d in text)
    # Clamp short-day hits to avoid over-counting (mon/tue etc. can appear in other docs)
    day_hits   = min(day_hits, 7)
    if day_hits >= 5:
        score += 28
        notes.append(f"{day_hits}_weekdays")
    elif day_hits >= 3:
        score += 14

    # Time values (HH:MM or military HHMM)
    time_hits = len(TIME_RE.findall(text))
    score    += min(time_hits * 4, 30)
    if time_hits:
        notes.append(f"{time_hits}_times")

    # Date values — low cap (dates appear in invoices too)
    date_hits = len(DATE_RE.findall(text))
    score    += min(date_hits * 2, 10)

    # "hours" keyword
    if HOURS_RE.search(text):
        score += 12
        notes.append("hours_field")

    # Weak positive words
    weak = {"employee":6,"department":5,"shift":8,"signature":7,
            "period":5,"traveler":10,"weekly":8,"certify":10,"contractor":8}
    score += min(sum(pts for kw, pts in weak.items() if kw in text), 22)

    # Row bonus (deferred — applied only after rejection checks)
    if   row_count >= 15: raw_row_bonus = 20
    elif row_count >= 8:  raw_row_bonus = 14
    elif row_count >= 5:  raw_row_bonus = 8
    else:                 raw_row_bonus = 0

    # ── PASS 2: Hard rejection ────────────────────────────────────────────

    reject = 0
    reject_notes: list[str] = []

    for kw, pts in HARD_REJECT_PHRASES.items():
        if kw in text:
            reject += pts
            reject_notes.append(f"-{pts}:{kw}")

    # Progress % values → NOT a timesheet signal
    pct_hits = len(PERCENT_RE.findall(text))
    if pct_hits >= 3:
        reject += 22
        reject_notes.append(f"-22:%x{pct_hits}")
    elif pct_hits >= 1:
        reject += 8

    # "start date" + "end date" PAIR = project table, not timesheet
    # (Note: "end date" alone is NOT penalised — "Week's End Date" would trigger it)
    if "start date" in text and "end date" in text:
        reject += 28
        reject_notes.append("-28:start+end_date_pair")

    if reject_notes:
        notes.extend(reject_notes[:4])

    # Apply rejection — preserve genuine timesheets with incidental bad words
    if reject > 0:
        if score > 65 and reject < 35:
            score = max(0, score - reject // 2)   # partial penalty for strong TS
        else:
            score = max(0, score - reject)

    # ── PASS 3: Soft rejection + column-header pattern + row bonus ────────

    soft = sum(pts for kw, pts in SOFT_REJECT_PHRASES.items() if kw in text)
    if score < 65:
        score = max(0, score - soft // 2)

    # Row bonus only if score is still timesheet-like
    if score >= 20:
        score += raw_row_bonus

    # Column-header pattern analysis
    ts_hdrs  = {"in","out","break","lunch","regular","overtime","reg","ot",
                "time in","time out","clock in","clock out",
                "shift start","shift end","lunch out","lunch in"}
    prj_hdrs = {"task","project","assigned","progress","status","priority",
                "milestone","sprint","start date","end date","due","days required"}

    ts_h  = sum(1 for w in ts_hdrs  if w in text)
    prj_h = sum(1 for w in prj_hdrs if w in text)

    if prj_h >= 3 and prj_h > ts_h:
        penalty = (prj_h - ts_h) * 8
        score   = max(0, score - penalty)
        notes.append(f"prj_headers={prj_h}")

    # Filename signal
    fn_pts, fn_note = filename_score(path)
    score += fn_pts
    if fn_note:
        notes.append(fn_note)

    return min(score, 100), ", ".join(notes[:8]) if notes else "no strong signals"


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
    # ── KEY FIX: actually rotate the image before OCR ─────────────────────
    gray = correct_rotation(gray)

    text, rows = ocr_extract(gray)
    score, notes = score_document(text, rows, path)
    is_ts, label = classify(score)
    return Result(path, is_ts, label, score, min(100, score), notes)


# ═════════════════════════════════════════════════════════════════════════════
#  FILE PROCESSORS
# ═════════════════════════════════════════════════════════════════════════════

def process_file(path: str) -> Result:
    ext = Path(path).suffix.lower()

    if ext in IMAGE_EXTS:
        gray = load_gray(path)
        if gray is None:
            return Result(path, False, "UNREADABLE", 0, 0, "could not read image")
        r = analyse_gray(gray, path)
        del gray; gc.collect()
        return r

    if ext in PDF_EXTS:
        gray = pdf_first_page_gray(path)
        if gray is None:
            if not _PDF:
                return Result(path, False, "PDF SKIPPED", 0, 0,
                              "install pymupdf: pip install pymupdf")
            return Result(path, False, "PDF ERROR", 0, 0, "could not render PDF")
        r = analyse_gray(gray, path)
        del gray; gc.collect()
        return r

    return Result(path, False, "UNSUPPORTED", 0, 0, ext)


# ═════════════════════════════════════════════════════════════════════════════
#  FOLDER SCANNING
# ═════════════════════════════════════════════════════════════════════════════

def _is_skipped(path: Path, root: Path) -> bool:
    try:
        return any(p in SKIP_DIRS for p in path.relative_to(root).parts[:-1])
    except ValueError:
        return True


def collect_files(folder: Path, recursive: bool) -> list[Path]:
    gen = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in gen
        if p.is_file()
        and p.suffix.lower() in ALL_EXTS
        and not p.name.startswith(".")
        and not (recursive and _is_skipped(p, folder))
    )


def default_folder() -> Path:
    if not TIMESHEET_FOLDER.is_dir():
        log.error("Folder not found: %s", TIMESHEET_FOLDER)
        sys.exit(1)
    return TIMESHEET_FOLDER


def warn_suspicious(folder: Path) -> None:
    s = str(folder).lower()
    if any(x in s for x in ("microsoft vs code","program files","appdata\\local\\programs")):
        log.error(
            "You are scanning a system folder, not your timesheet project.\n"
            "  Scanned: %s\n  Use:     python \"%s\" --folder \"%s\"",
            folder, PROJECT_DIR / "detector.py", PROJECT_DIR,
        )
        sys.exit(1)


def scan_folder(folder: str, recursive: bool = False) -> list[Result]:
    fp = Path(folder).resolve()
    if not fp.is_dir():
        log.error("Folder not found: %s", fp)
        sys.exit(1)

    warn_suspicious(fp)
    files = collect_files(fp, recursive)
    if not files:
        log.warning("No image/PDF files found in %s", fp)
        return []

    log.info("\n%s", "=" * 70)
    log.info("  TIMESHEET DETECTOR  —  %d file(s) in  %s", len(files), fp)
    log.info("%s\n", "=" * 70)

    results: list[Result] = []
    for i, fpath in enumerate(files, 1):
        log.info("[%d/%d]  %s", i, len(files), fpath.name)
        r = process_file(str(fpath))
        log.info("%s", r.line())
        if r.notes:
            log.info("         notes: %s", r.notes)
        log.info("")
        results.append(r)

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY & CSV
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[Result]) -> None:
    if not results:
        return

    yes    = [r for r in results if r.is_timesheet]
    review = [r for r in results if r.label == "LIKELY (review manually)"]
    no     = [r for r in results
              if not r.is_timesheet and r.label not in
              ("UNREADABLE","PDF ERROR","PDF SKIPPED","UNSUPPORTED")]

    log.info("%s", "=" * 70)
    log.info("  SUMMARY")
    log.info("  Total scanned : %d", len(results))
    log.info("  ✅ Timesheets  : %d", len(yes))
    log.info("  🟡 Review      : %d", len(review))
    log.info("  ❌ Not TS      : %d", len(no))
    log.info("%s", "=" * 70)

    if yes:
        log.info("\n  CONFIRMED TIMESHEETS:")
        for r in yes:
            log.info("    • %s  (%d%%)", Path(r.path).name, r.confidence)
    if review:
        log.info("\n  REVIEW MANUALLY:")
        for r in review:
            log.info("    • %s  (%d%%)", Path(r.path).name, r.confidence)
    if not yes and not review:
        log.info("\n  No timesheets found.")
    log.info("")


def save_csv(results: list[Result], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file","is_timesheet","label","score","confidence","notes","full_path"])
        for r in results:
            w.writerow([Path(r.path).name, "YES" if r.is_timesheet else "NO",
                        r.label, r.score, r.confidence, r.notes, r.path])
    log.info("Results saved to: %s\n", out_path)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Detect timesheets in a folder")
    ap.add_argument("--folder",    default="",   help=f"Folder to scan (default: {PROJECT_DIR})")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out",       default="",   help="CSV output path")
    ap.add_argument("--verbose",   action="store_true", help="Show debug info")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not _OCR:
        log.error("pytesseract not installed. Run: pip install pytesseract")
        sys.exit(1)

    folder = Path(args.folder).resolve() if args.folder else default_folder()
    log.info("\n*** Scanning: %s ***\n", folder)

    results = scan_folder(str(folder), recursive=args.recursive)
    print_summary(results)

    out = args.out or str(PROJECT_DIR / "timesheet_results.csv")
    save_csv(results, out)


if __name__ == "__main__":
    main()