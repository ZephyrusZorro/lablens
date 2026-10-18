"""Datasheet auto-scraper: unknown part number -> search -> download the PDF -> hand it to the store.

Polite by design: identifies itself, rate-limits, caches misses for a day, caps file size,
and only downloads PDFs (never executes or renders remote content).
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urljoin

log = logging.getLogger(__name__)
UA = "LabLens/0.4 (student lab assistant; contact: your-lab-instructor)"
PDF_LINK = re.compile(r'href="([^"]+\.pdf[^"]*)"', re.I)
RESULT_LINK = re.compile(r'class="result__a"[^>]*href="([^"]+)"', re.I)


class DatasheetScraper:
    def __init__(self, download_dir: Path, max_mb: float = 15, min_interval: float = 2.0, miss_ttl: float = 86400):
        self.dir = Path(download_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.min_interval = min_interval
        self.miss_ttl = miss_ttl
        self._last = 0.0
        self._misses: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg) -> "DatasheetScraper":
        return cls(cfg.path("scraper.download_dir", "data/datasheets/scraped"), cfg.get("scraper.max_mb", 15))

    def _sleep(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str, **kw):
        import requests

        self._sleep()
        return requests.get(url, headers={"User-Agent": UA}, timeout=15, **kw)

    def _candidate_pdfs(self, part: str) -> list[str]:
        q = quote_plus(f"{part} datasheet filetype:pdf")
        html = self._get(f"https://html.duckduckgo.com/html/?q={q}").text
        pages = [m for m in RESULT_LINK.findall(html)]
        pdfs = [m for m in PDF_LINK.findall(html)]
        # Also look one hop into result pages that aren't PDFs (manufacturer product pages)
        for u in pages[:3]:
            if u.lower().endswith(".pdf"):
                pdfs.append(u)
        return [urljoin("https:", p) if p.startswith("//") else p for p in pdfs]

    def fetch(self, part: str) -> Optional[Path]:
        key = part.lower()
        if key in self._misses and time.time() - self._misses[key] < self.miss_ttl:
            return None
        local = self.dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', part).strip('_')}_datasheet.pdf"
        if local.exists():
            return local
        for url in self._candidate_pdfs(part)[:5]:
            try:
                r = self._get(url, stream=True)
                if not r.ok or "pdf" not in r.headers.get("content-type", "").lower():
                    continue
                size = 0
                with open(local, "wb") as f:
                    for chunk in r.iter_content(64 * 1024):
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError("file too large")
                        f.write(chunk)
                if not local.read_bytes()[:5] == b"%PDF-":
                    local.unlink(missing_ok=True)
                    continue
                log.info("downloaded %s from %s", part, url)
                return local
            except Exception as e:
                log.info("skip %s: %s", url, e)
                local.unlink(missing_ok=True)
        self._misses[key] = time.time()
        return None
