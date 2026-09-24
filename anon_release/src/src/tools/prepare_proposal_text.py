# -*- coding: utf-8 -*-
"""
Stage 0: Text Preparation · prepare_proposal_text.py

Features:
- Automatically select the latest proposal file from src/data/proposals/ (or specify manually using --file)
- Accepts a proposal file (currently supports: PDF / DOCX / PPTX / TXT / MD)
- Extracts text from each page/entire document as completely as possible
- For PDFs:
    - First attempts text extraction with pdfplumber
    - Resets to OCR (pytesseract + pdf2image) automatically if a page has minimal text
- Outputs:
    - src/data/prepared/<proposal_id>/full_text.txt
    - src/data/prepared/<proposal_id>/pages.json (page-by-page text + whether OCR was used + global offsets)
"""

import os
import json
import argparse
import shutil
import time
import sys
from pathlib import Path

import base64
from io import BytesIO

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import active_provider, default_model, vision_completion

load_dotenv()


# ========== Configuration ==========

MIN_TEXT_CHARS_PER_PAGE = 30  # Try OCR if character count goes below this threshold
LOW_TEXT_WARN_CHARS = 100      # Flag limits below this character count for manual review
PREVIEW_CHARS = int(os.getenv("STAGE0_PREVIEW_CHARS", "260"))
SUPPORTED_PROPOSAL_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt"]

_PROGRESS_FILE = Path(__file__).resolve().parents[2] / "src" / "data" / "step_progress.json"

def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = _PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else _PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass

# You can add chi_sim if you have Chinese OCR data
# Common configurations: "eng" / "chi_sim" / "chi_sim+eng"
TESSERACT_LANG = os.getenv("TESS_LANG", "chi_sim+eng")
PROVIDER       = active_provider()
VISION_MODEL   = os.getenv("VISION_MODEL", default_model(PROVIDER))
ENABLE_VISION  = os.getenv("ENABLE_VISION", "true").strip().lower() == "true"
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
VISION_FAILURE_DISABLE_AFTER = int(os.getenv("VISION_FAILURE_DISABLE_AFTER", "3"))
_VISION_CONSECUTIVE_FAILURES = 0
_VISION_DISABLED_FOR_RUN = False


# ========== Diagnostic Logs ==========

def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def _preview_text(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _source_counts(page_sources: list[str]) -> dict:
    counts = {}
    for src in page_sources:
        counts[src] = counts.get(src, 0) + 1
    return counts


def _choose_best_short_pdf_text(pdf_text: str, ocr_text: str) -> tuple[str, str]:
    """
    Keep the most useful fallback text for low-text PDF pages.

    A short native PDF extraction can still contain important labels, headings, or
    page numbers. If OCR is empty or clearly shorter, preserving native text is
    better than replacing it with weaker output.
    """
    pdf_text = (pdf_text or "").strip()
    ocr_text = (ocr_text or "").strip()
    if ocr_text and len(ocr_text) >= len(pdf_text):
        return ocr_text, "ocr"
    if pdf_text:
        return pdf_text, "pdf_text_short"
    if ocr_text:
        return ocr_text, "ocr"
    return "", "empty"


def _print_runtime_diagnostics(file_path: Path, file_type: str, use_ocr: bool) -> None:
    size_mb = file_path.stat().st_size / (1024 * 1024)
    _log("STAGE0", "Starting document text extraction")
    _log("STAGE0", f"input_file={file_path.resolve()}")
    _log("STAGE0", f"file_type={file_type} size_mb={size_mb:.2f}")
    _log("STAGE0", f"use_ocr={use_ocr} enable_vision={ENABLE_VISION}")
    _log("STAGE0", f"tesseract_lang={TESSERACT_LANG} vision_model={VISION_MODEL}")
    if file_type == "pdf":
        _log("DIAG", f"pdftoppm_path={shutil.which('pdftoppm') or '<not found>'}")
        _log("DIAG", f"tesseract_path={shutil.which('tesseract') or '<not found>'}")
        if use_ocr and not shutil.which("tesseract"):
            _log("WARN", "OCR is enabled but tesseract was not found on PATH.")
        if not shutil.which("pdftoppm"):
            _log("WARN", "Poppler pdftoppm was not found on PATH; pdf2image/convert_from_path may fail.")


def _build_page_audit(pages_text: list[str], page_sources: list[str]) -> list[dict]:
    audit = []
    for idx, (text, source) in enumerate(zip(pages_text, page_sources), start=1):
        stripped = (text or "").strip()
        audit.append(
            {
                "page_index": idx,
                "source": source,
                "char_len": len(stripped),
                "line_count": len([ln for ln in stripped.splitlines() if ln.strip()]),
                "has_visual_content_note": "[VISUAL CONTENT:" in stripped,
                "needs_review": len(stripped) < LOW_TEXT_WARN_CHARS,
                "preview": _preview_text(stripped),
            }
        )
    return audit


def _print_extraction_summary(pages_text: list[str], page_sources: list[str], elapsed_sec: float) -> None:
    page_count = len(pages_text)
    char_lengths = [len((txt or "").strip()) for txt in pages_text]
    total_chars = sum(char_lengths)
    empty_pages = [i + 1 for i, n in enumerate(char_lengths) if n == 0]
    low_text_pages = [i + 1 for i, n in enumerate(char_lengths) if 0 < n < LOW_TEXT_WARN_CHARS]
    visual_pages = [i + 1 for i, txt in enumerate(pages_text) if "[VISUAL CONTENT:" in (txt or "")]

    _log("SUMMARY", f"pages_or_units={page_count} total_chars={total_chars} elapsed_sec={elapsed_sec:.2f}")
    _log("SUMMARY", f"source_counts={json.dumps(_source_counts(page_sources), ensure_ascii=False)}")
    if char_lengths:
        _log(
            "SUMMARY",
            f"chars_per_page min={min(char_lengths)} max={max(char_lengths)} "
            f"avg={total_chars / max(page_count, 1):.1f}"
        )
    if empty_pages:
        _log("WARN", f"empty_pages={empty_pages}")
    if low_text_pages:
        _log("WARN", f"low_text_pages_under_{LOW_TEXT_WARN_CHARS}_chars={low_text_pages}")
    if visual_pages:
        _log("SUMMARY", f"pages_with_visual_descriptions={visual_pages}")


# ========== File Type Detection ==========

def detect_file_type(path: Path) -> str:
    """
    Determine file type simply based on its extension.
    Returns: "pdf" / "docx" / "pptx" / "txt"
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in [".docx", ".doc"]:
        return "docx"
    if suffix in [".txt", ".md"]:
        return "txt"
    if suffix in [".pptx", ".ppt"]:
        return "pptx"
    raise ValueError(f"Unsupported file type: {suffix}")


def find_latest_proposal() -> Path:
    """
    Automatically search for the most "recently modified" proposal file in src/data/proposals/.
    Supported extensions: .pdf / .docx / .doc / .txt / .md / .pptx / .ppt
    """
    base_dir = Path(__file__).resolve().parents[2]
    proposals_dir = base_dir / "src" / "data" / "proposals"

    if not proposals_dir.exists():
        raise FileNotFoundError(f"Proposal directory not found: {proposals_dir}")

    candidates = []
    for p in proposals_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_PROPOSAL_EXTENSIONS:
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No available files found in directory: {proposals_dir}")

    latest = max(candidates, key=lambda x: x.stat().st_mtime)
    print(f"[INFO] [auto] Selected latest proposal file: {latest}")
    return latest


# ========== Extraction Functions ==========

def ocr_page_from_pdf(pdf_path: Path, page_index: int) -> str:
    """
    Perform OCR on a designated page using pdf2image + pytesseract.
    page_index: 0-based
    """
    start = time.perf_counter()
    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_index + 1,
            last_page=page_index + 1
        )
    except Exception as e:
        _log("WARN", f"convert_from_path failed page={page_index+1}: {e}")
        return ""

    if not images:
        _log("WARN", f"convert_from_path returned no images page={page_index+1}")
        return ""

    image: Image.Image = images[0]
    _log("DIAG", f"OCR image page={page_index+1} mode={image.mode} size={image.size}")

    try:
        text = pytesseract.image_to_string(image, lang=TESSERACT_LANG)
        _log(
            "DIAG",
            f"OCR finished page={page_index+1} chars={len((text or '').strip())} "
            f"elapsed_sec={time.perf_counter() - start:.2f}"
        )
        return text
    except Exception as e:
        _log("WARN", f"OCR failed page={page_index+1}: {e}")
        return ""


def describe_image_with_vision(pil_image, context_hint: str = "") -> str:
    """Send a PIL Image to the vision LLM; return a plain-text description or '' on any failure."""
    global _VISION_CONSECUTIVE_FAILURES, _VISION_DISABLED_FOR_RUN
    if not ENABLE_VISION:
        return ""
    if _VISION_DISABLED_FOR_RUN:
        return ""
    start = time.perf_counter()
    try:
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        prompt = (
            "You are analyzing a page from a business/research proposal document. "
            "Describe ALL visual content present: charts, graphs (bar/line/pie/scatter), "
            "tables, figures, diagrams, flowcharts. Include key numbers, percentages, "
            "axis labels, legends, and any visible trends or conclusions. "
            "Be concise but complete. Use plain text only. "
            "If there is no meaningful visual content, respond with exactly: "
            "no significant visual content."
        )
        if context_hint:
            prompt += f" Context: {context_hint}"
        resp = vision_completion(
            provider=PROVIDER,
            model=VISION_MODEL,
            prompt=prompt,
            image_b64=b64,
            mime_type="image/png",
            max_tokens=512,
        )
        desc = resp.content.strip()
        _VISION_CONSECUTIVE_FAILURES = 0
        _log(
            "DIAG",
            f"Vision call finished context={context_hint or '<none>'} "
            f"desc_chars={len(desc)} elapsed_sec={time.perf_counter() - start:.2f}"
        )
        return "" if desc.lower().startswith("no significant") else desc
    except Exception as e:
        _VISION_CONSECUTIVE_FAILURES += 1
        _log("WARN", f"Vision LLM call failed context={context_hint or '<none>'}: {e}")
        if _VISION_CONSECUTIVE_FAILURES >= VISION_FAILURE_DISABLE_AFTER:
            _VISION_DISABLED_FOR_RUN = True
            _log(
                "WARN",
                f"Disabling further vision calls for this document after "
                f"{_VISION_CONSECUTIVE_FAILURES} consecutive failures."
            )
        return ""


def extract_from_pdf(pdf_path: Path, use_ocr: bool = True, pid: str = ""):
    """
    Extract text page by page from PDF:
    - First attempts text extraction with pdfplumber
    - If a page has minimal text and use_ocr=True, it resorts to OCR

    Returns:
      pages_text: list[str]   Text of each page (order matching pages)
      page_sources: list[str] Source markers per page: "pdf_text" / "ocr" / "empty"
    """
    pages_text = []
    page_sources = []

    pdf_path = Path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        _log("INFO", f"PDF pages={num_pages}")
        _write_progress(0, num_pages, pid)
        for i, page in enumerate(pdf.pages):
            page_start = time.perf_counter()
            txt = page.extract_text() or ""
            txt = txt.strip()
            image_count = len(page.images or [])
            _log(
                "PAGE",
                f"page={i+1}/{num_pages} width={page.width:.1f} height={page.height:.1f} "
                f"pdfplumber_chars={len(txt)} image_count={image_count}"
            )

            if txt and len(txt) >= MIN_TEXT_CHARS_PER_PAGE:
                pages_text.append(txt)
                page_sources.append("pdf_text")
                _log("PAGE", f"page={i+1} source=pdf_text chars={len(txt)}")
            else:
                if use_ocr:
                    _log("PAGE", f"page={i+1} pdf text too short ({len(txt)} chars); trying OCR")
                    ocr_txt = ocr_page_from_pdf(pdf_path, page_index=i)
                    best_txt, source = _choose_best_short_pdf_text(txt, ocr_txt)
                    pages_text.append(best_txt)
                    page_sources.append(source)
                    if source == "ocr":
                        _log("PAGE", f"page={i+1} source=ocr chars={len(best_txt)}")
                    elif source == "pdf_text_short":
                        _log(
                            "PAGE",
                            f"page={i+1} keeping short pdf text chars={len(best_txt)} "
                            f"ocr_chars={len((ocr_txt or '').strip())}"
                        )
                    else:
                        _log("WARN", f"page={i+1} OCR returned empty text")
                else:
                    pages_text.append(txt)
                    page_sources.append("pdf_text_empty")
                    _log("PAGE", f"page={i+1} OCR disabled; keeping short pdf text chars={len(txt)}")
            _write_progress(i + 1, num_pages, pid)

            # Vision: describe embedded images/charts on this page
            if ENABLE_VISION and not _VISION_DISABLED_FOR_RUN and page.images:
                _log("PAGE", f"page={i+1} found {len(page.images)} image(s); calling vision model")
                try:
                    page_imgs = convert_from_path(
                        str(pdf_path), first_page=i + 1, last_page=i + 1, dpi=150
                    )
                    if page_imgs:
                        visual_desc = describe_image_with_vision(
                            page_imgs[0], context_hint=f"Page {i+1} of proposal"
                        )
                        if visual_desc:
                            pages_text[-1] = pages_text[-1] + f"\n[VISUAL CONTENT: {visual_desc}]"
                            _log("PAGE", f"page={i+1} appended visual description chars={len(visual_desc)}")
                        else:
                            _log("PAGE", f"page={i+1} vision returned no significant visual content")
                except Exception as e:
                    _log("WARN", f"page={i+1} visual analysis failed: {e}")

            final_chars = len((pages_text[-1] if pages_text else "").strip())
            if final_chars < LOW_TEXT_WARN_CHARS:
                _log("WARN", f"page={i+1} final text is short ({final_chars} chars); inspect pages.json")
            _log("PREVIEW", f"page={i+1} text_preview={_preview_text(pages_text[-1] if pages_text else '')}")
            _log("PAGE", f"page={i+1} done elapsed_sec={time.perf_counter() - page_start:.2f}")

    return pages_text, page_sources


def extract_from_docx(docx_path: Path):
    """
    Extract text from DOCX and describe embedded images with vision model.
    Treated as a single continuous page.
    """
    doc = Document(str(docx_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    table_blocks = []
    for table_idx, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            row_text = " | ".join(cell for cell in cells if cell)
            if row_text:
                rows.append(row_text)
        if rows:
            table_blocks.append(f"[TABLE {table_idx}]\n" + "\n".join(rows))

    text_parts = paragraphs + table_blocks
    text = "\n".join(text_parts)
    _log(
        "INFO",
        f"DOCX paragraphs={len(paragraphs)} tables={len(doc.tables)} "
        f"table_blocks={len(table_blocks)} inline_images={len(doc.inline_shapes)} text_chars={len(text)}"
    )

    if ENABLE_VISION and not _VISION_DISABLED_FOR_RUN and doc.inline_shapes:
        visual_parts = []
        for idx, shape in enumerate(doc.inline_shapes):
            try:
                rId = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
                blob = doc.part.related_parts[rId].blob
                pil_image = Image.open(BytesIO(blob))
                desc = describe_image_with_vision(
                    pil_image, context_hint=f"Inline image {idx+1} from DOCX proposal"
                )
                if desc:
                    visual_parts.append(f"[VISUAL CONTENT: {desc}]")
                    _log("INFO", f"DOCX image={idx+1} appended visual description chars={len(desc)}")
                else:
                    _log("INFO", f"DOCX image={idx+1} returned no significant visual content")
            except Exception as e:
                _log("WARN", f"DOCX image={idx+1} processing failed: {e}")
        if visual_parts:
            text = text + "\n" + "\n".join(visual_parts)

    _log("PREVIEW", f"DOCX text_preview={_preview_text(text)}")
    return [text], ["docx"]


def _extract_pptx_table_text(shape) -> list[str]:
    if not getattr(shape, "has_table", False):
        return []
    rows = []
    for row in shape.table.rows:
        cells = [" ".join(cell.text.split()) for cell in row.cells]
        row_text = " | ".join(cell for cell in cells if cell)
        if row_text:
            rows.append(row_text)
    return rows


def extract_from_pptx(pptx_path: Path, pid: str = ""):
    """
    Extract text slide by slide from PPTX, and invoke vision model for picture shapes.
    Each slide is treated as a single page.
    """
    prs = Presentation(str(pptx_path))
    pages_text = []
    page_sources = []

    num_slides = len(prs.slides)
    _log("INFO", f"PPTX slides={num_slides}")
    _write_progress(0, num_slides, pid)

    for slide_idx, slide in enumerate(prs.slides):
        slide_start = time.perf_counter()
        slide_texts = []
        visual_parts = []
        picture_count = 0
        text_shape_count = 0
        table_count = 0

        for shape in slide.shapes:
            # Extract text from any shape with a text frame
            if shape.has_text_frame:
                text_shape_count += 1
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        slide_texts.append(line)

            table_rows = _extract_pptx_table_text(shape)
            if table_rows:
                table_count += 1
                slide_texts.append(f"[TABLE {table_count}]")
                slide_texts.extend(table_rows)

            # Extract embedded pictures for vision analysis
            if ENABLE_VISION and not _VISION_DISABLED_FOR_RUN:
                try:
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        picture_count += 1
                        blob = shape.image.blob
                        pil_image = Image.open(BytesIO(blob))
                        desc = describe_image_with_vision(
                            pil_image,
                            context_hint=f"Slide {slide_idx + 1} image from PowerPoint proposal"
                        )
                        if desc:
                            visual_parts.append(f"[VISUAL CONTENT: {desc}]")
                            _log("INFO", f"slide={slide_idx+1} image appended visual description chars={len(desc)}")
                        else:
                            _log("INFO", f"slide={slide_idx+1} image returned no significant visual content")
                except Exception as e:
                    _log("WARN", f"slide={slide_idx+1} image processing failed: {e}")

        slide_text = "\n".join(slide_texts)
        if visual_parts:
            slide_text = slide_text + "\n" + "\n".join(visual_parts)

        pages_text.append(slide_text)
        page_sources.append("pptx")
        _log(
            "PAGE",
            f"slide={slide_idx+1}/{num_slides} text_shapes={text_shape_count} "
            f"tables={table_count} pictures={picture_count} chars={len(slide_text)} "
            f"elapsed_sec={time.perf_counter() - slide_start:.2f}"
        )
        if len(slide_text.strip()) < LOW_TEXT_WARN_CHARS:
            _log("WARN", f"slide={slide_idx+1} final text is short ({len(slide_text.strip())} chars)")
        _log("PREVIEW", f"slide={slide_idx+1} text_preview={_preview_text(slide_text)}")
        _write_progress(slide_idx + 1, num_slides, pid)

    return pages_text, page_sources


def extract_from_txt(txt_path: Path):
    """
    Extract from raw text file.
    Treated as a single continuous page.
    """
    content = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    content = content.strip()
    _log("INFO", f"TXT chars={len(content)}")
    _log("PREVIEW", f"TXT text_preview={_preview_text(content)}")
    return [content], ["txt"]


def prepare_text(file_path: Path, proposal_id: str, use_ocr: bool = True):
    """
    Core entry point:
    - Identify file type
    - Invoke corresponding extraction function
    - Output full_text.txt + pages.json
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    stage_start = time.perf_counter()
    file_type = detect_file_type(file_path)
    _print_runtime_diagnostics(file_path, file_type, use_ocr)

    if file_type == "pdf":
        pages_text, page_sources = extract_from_pdf(file_path, use_ocr=use_ocr, pid=proposal_id)
    elif file_type == "docx":
        pages_text, page_sources = extract_from_docx(file_path)
    elif file_type == "pptx":
        pages_text, page_sources = extract_from_pptx(file_path, pid=proposal_id)
    elif file_type == "txt":
        pages_text, page_sources = extract_from_txt(file_path)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    full_text = "\n\n".join(pages_text)
    elapsed_sec = time.perf_counter() - stage_start
    _print_extraction_summary(pages_text, page_sources, elapsed_sec)

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "src" / "data" / "prepared" / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full_text.txt
    full_text_path = out_dir / "full_text.txt"
    full_text_path.write_text(full_text, encoding="utf-8")
    _log("OK", f"full_text.txt written: {full_text_path} chars={len(full_text)}")

    # Save pages.json, adding global_char_start/global_char_end for global offset tracking
    pages_data = []
    offset = 0
    for i, (txt, src) in enumerate(zip(pages_text, page_sources)):
        char_len = len(txt)
        page_start = offset
        page_end = offset + char_len
        pages_data.append(
            {
                "page_index": i + 1,
                "source": src,
                "char_len": char_len,
                "global_char_start": page_start,
                "global_char_end": page_end,
                "text": txt,
            }
        )
        # Using "\n\n" to join full_text means we add 2 characters for the delimiter length
        offset = page_end + 2

    pages_json_path = out_dir / "pages.json"
    pages_json_path.write_text(json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("OK", f"pages.json written: {pages_json_path}")

    audit = {
        "proposal_id": proposal_id,
        "input_file": str(file_path.resolve()),
        "file_type": file_type,
        "use_ocr": use_ocr,
        "enable_vision": ENABLE_VISION,
        "tesseract_lang": TESSERACT_LANG,
        "vision_model": VISION_MODEL,
        "elapsed_sec": round(elapsed_sec, 2),
        "source_counts": _source_counts(page_sources),
        "total_chars": len(full_text),
        "pages": _build_page_audit(pages_text, page_sources),
    }
    audit_path = out_dir / "stage0_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("OK", f"stage0_audit.json written: {audit_path}")

    return {
        "proposal_id": proposal_id,
        "file_type": file_type,
        "out_dir": str(out_dir),
        "full_text_path": str(full_text_path),
        "pages_json_path": str(pages_json_path),
        "audit_path": str(audit_path),
        "num_pages": len(pages_text),
        "total_chars": len(full_text),
        "source_counts": _source_counts(page_sources),
    }


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="Stage 0: Proposal Text Preparation (w/ OCR + auto-selection)")
    parser.add_argument("--file", required=False, help="Proposal file path (PDF/DOCX/PPTX/TXT/MD). If empty, auto-selects the latest.")
    parser.add_argument("--proposal_id", required=False, help="Proposal ID (used for output dir name; defaults to file stem if omitted).")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR (for fast debugging mainly).")
    args = parser.parse_args()

    # 1) Determine file_path
    if args.file:
        file_path = Path(args.file)
        _log("INFO", f"Using user-provided file: {file_path}")
    else:
        file_path = find_latest_proposal()

    # 2) Determine proposal_id
    proposal_id = args.proposal_id or file_path.stem

    # 3) Check if OCR is enabled
    use_ocr = not args.no_ocr

    info = prepare_text(file_path, proposal_id, use_ocr=use_ocr)

    print("\n[SUMMARY]", flush=True)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
