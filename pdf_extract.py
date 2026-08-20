import io
import re
from dataclasses import dataclass

import pdfplumber


BULLET_GLYPHS = set("•●▪·◦●■□◆◇▶►▸∙⁃")

RESUME_SECTION_HINTS = [
    "summary",
    "objective",
    "experience",
    "work experience",
    "employment",
    "education",
    "skills",
    "technical skills",
    "projects",
    "publications",
    "certifications",
    "awards",
    "activities",
    "interests",
    "languages",
    "references",
]


@dataclass
class PDFDoc:
    text: str
    pages: int
    noise_ratio: float
    detected_sections: list[str]

    @property
    def full_text(self) -> str:
        return self.text


def extract_pdf(file_buffer) -> PDFDoc:
    """Read a PDF from an in-memory buffer. Never writes to disk."""
    data = file_buffer.read() if hasattr(file_buffer, "read") else file_buffer
    if hasattr(file_buffer, "seek"):
        try:
            file_buffer.seek(0)
        except Exception:
            pass

    text_pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_pages.append(t)

    raw = "\n".join(text_pages)
    cleaned = _dehyphenate(raw)
    noise = _noise_ratio(raw)
    detected = _detect_sections(cleaned)
    return PDFDoc(
        text=cleaned,
        pages=n_pages,
        noise_ratio=noise,
        detected_sections=detected,
    )


def _dehyphenate(text: str) -> str:
    # join lines that end with "word-" onto the next line's leading word
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _noise_ratio(text: str) -> float:
    """Fraction of characters that look like structural noise, not content."""
    if not text:
        return 0.0
    total = len(text)
    noise = 0
    # bullet glyphs
    for ch in text:
        if ch in BULLET_GLYPHS:
            noise += 1
    # per-line noise: page numbers, contact-only lines, all-caps banners
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r"page\s*\d+(\s*of\s*\d+)?", s.lower()):
            noise += len(line)
        elif re.fullmatch(r"\d{1,3}", s):
            noise += len(line)
        elif _looks_like_contact(s):
            noise += len(line)
        elif len(s) <= 40 and s.upper() == s and any(c.isalpha() for c in s):
            noise += len(line)
    return min(1.0, noise / total)


def _looks_like_contact(s: str) -> bool:
    has_email = "@" in s and re.search(r"\S+@\S+\.\S+", s) is not None
    has_phone = re.search(r"\+?\d[\d\s().-]{7,}\d", s) is not None
    has_url = re.search(r"https?://|linkedin\.com|github\.com", s.lower()) is not None
    return has_email or has_phone or has_url


def _detect_sections(text: str) -> list[str]:
    found: list[str] = []
    lc = text.lower()
    for hint in RESUME_SECTION_HINTS:
        # match as its own short line
        pattern = rf"(?m)^\s*{re.escape(hint)}\s*:?\s*$"
        if re.search(pattern, lc):
            found.append(hint)
    return found
