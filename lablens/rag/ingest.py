"""PDF -> page-aware chunks. Every chunk remembers its source file and page for citations."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

# Which detector labels a datasheet is about, guessed from its filename.
# Extend data/datasheets/parts.yaml rather than editing this.
DEFAULT_PART_KEYWORDS: Dict[str, List[str]] = {
    "ic": ["555", "ne555", "lm358", "lm741", "74hc", "cd40", "atmega", "l293", "uln2003", "ic-", "_ic"],
    "led": ["led"],
    "resistor": ["resistor", "colour", "color-code"],
    "capacitor": ["capacitor", "electrolytic", "ceramic"],
    "diode": ["diode", "1n400", "1n4148", "zener"],
    "transistor": ["transistor", "bc547", "2n2222", "2n3904", "tip120"],
    "arduino": ["arduino", "nano", "uno"],
    "breadboard": ["breadboard"],
    "potentiometer": ["potentiometer", "trimpot"],
    "button": ["button", "switch", "tactile"],
    "manual": ["manual", "lab", "experiment"],
}


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int  # 1-based, as printed in a PDF viewer
    tags: List[str] = field(default_factory=list)
    topic: str = ""  # primary component this page is about (first one named in its heading)

    def to_dict(self):
        return asdict(self)


def load_part_keywords(pdf_dir: Path) -> Dict[str, List[str]]:
    kw = {k: list(v) for k, v in DEFAULT_PART_KEYWORDS.items()}
    f = pdf_dir / "parts.yaml"
    if f.exists():
        import yaml

        for k, v in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).items():
            kw.setdefault(k, [])
            kw[k].extend(str(x).lower() for x in v)
    return kw


def tags_for(filename: str, keywords: Dict[str, List[str]]) -> List[str]:
    name = filename.lower()
    return sorted({label for label, words in keywords.items() if any(w in name for w in words)})


# Content tagging: which detector labels a passage is *about*, judged from its opening text
# (page heading + first lines). Word-boundary matches avoid "led" in "called".
CONTENT_KEYWORDS: Dict[str, List[str]] = {
    "ic": ["ne555", "555", "lm358", "op-amp", "integrated circuit", "dip ic", "timer ic"],
    "led": ["led", "leds"],
    "resistor": ["resistor", "resistors", "colour code", "color code"],
    "capacitor": ["capacitor", "capacitors", "electrolytic"],
    "diode": ["diode", "diodes", "zener", "1n4007", "1n4148"],
    "transistor": ["transistor", "transistors", "bc547", "2n2222", "npn"],
    "arduino": ["arduino", "nano", "uno", "atmega328p"],
    "breadboard": ["breadboard"],
}
_CONTENT_RE = {k: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\b", re.I) for k, v in CONTENT_KEYWORDS.items()}


def content_tags(text: str, head_chars: int = 220) -> List[str]:
    head = text[:head_chars]
    return sorted(k for k, rx in _CONTENT_RE.items() if rx.search(head))


def primary_topic(text: str) -> str:
    """Label of the component named earliest in the first line (the page heading)."""
    first = text.strip().split("\n", 1)[0][:120]
    hits = [(m.start(), k) for k, rx in _CONTENT_RE.items() for m in [rx.search(first)] if m]
    return min(hits)[1] if hits else ""


def strip_boilerplate(pages: List[str], min_pages: int = 3, frac: float = 0.5) -> List[str]:
    """Remove running headers/footers: lines that recur (digits normalised) on most pages."""
    if len(pages) < min_pages:
        return pages
    norm = lambda ln: re.sub(r"\d+", "#", ln.strip().lower())
    counts: Dict[str, int] = {}
    for p in pages:
        for ln in {norm(x) for x in p.splitlines() if x.strip()}:
            counts[ln] = counts.get(ln, 0) + 1
    boiler = {ln for ln, n in counts.items() if n >= max(min_pages, frac * len(pages))}
    return ["\n".join(x for x in p.splitlines() if norm(x) not in boiler) for p in pages]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):  # try to break on a paragraph / sentence boundary
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size * 0.5:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def pdf_to_chunks(path: Path, size: int = 900, overlap: int = 150, keywords=None) -> List[Chunk]:
    from pypdf import PdfReader

    keywords = keywords or DEFAULT_PART_KEYWORDS
    reader = PdfReader(str(path))
    tags = tags_for(path.name, keywords)
    raw = []
    for page in reader.pages:
        try:
            raw.append(page.extract_text() or "")
        except Exception:
            raw.append("")
    chunks: List[Chunk] = []
    for pno, text in enumerate(strip_boilerplate(raw), start=1):
        text = _clean(text)
        if not text:
            continue
        page_tags = content_tags(text)  # page heading decides the topic of every chunk on the page
        topic = primary_topic(text)
        for i, c in enumerate(chunk_text(text, size, overlap)):
            t = sorted(set(tags) | set(page_tags) | set(content_tags(c)))
            chunks.append(Chunk(f"{path.stem}:{pno}:{i}", c, path.name, pno, t, topic))
    return chunks


def text_to_chunks(text: str, source: str, page: int = 1, tags=None, size=900, overlap=150) -> List[Chunk]:
    """For non-PDF sources (scraped HTML pages, notes)."""
    text = _clean(text)
    topic = primary_topic(text)
    return [Chunk(f"{Path(source).stem}:{page}:{i}", c, source, page, sorted(set(tags or []) | set(content_tags(c))),
                  topic) for i, c in enumerate(chunk_text(text, size, overlap))]
