"""Hybrid retrieval: BM25 (inverted index) + optional dense embeddings, fused with RRF and boosted by
the component the student is pointing at.

* Sparse: BM25 over an inverted index - a query touches only the postings of its own terms, so cost
  scales with matches, not corpus size. Catches exact tokens ("NE555", "pin 4") that embeddings blur.
* Dense (optional): sentence-transformers if installed; FAISS inner-product index if installed, numpy otherwise.
* Fusion: reciprocal-rank fusion, then +boost for chunks tagged with the pointed component / part number.
* Incremental: add_pdf()/add_text() append without a rebuild (used by the datasheet scraper).
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lablens.rag.ingest import Chunk, load_part_keywords, pdf_to_chunks, text_to_chunks
from lablens.types import Citation

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset("a an the is are was were be to of in on for and or with it its this that what which "
                  "where how do does i my me you your can should would will at by as from".split())


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


# Bench vocabulary -> datasheet vocabulary (applied to queries only)
SYNONYMS = {
    "leg": ["pin", "lead"], "legs": ["pins", "leads"], "lead": ["pin", "leg"], "ground": ["gnd"],
    "gnd": ["ground"], "negative": ["cathode", "minus"], "positive": ["anode", "plus"], "plus": ["positive"],
    "minus": ["negative"], "vcc": ["supply", "power"], "power": ["vcc", "supply"], "supply": ["vcc"],
    "collector": ["pins", "c"], "emitter": ["pins", "e"], "base": ["pins", "b"], "value": ["colour", "bands"],
    "colour": ["color", "bands"], "color": ["colour", "bands"], "blink": ["astable", "frequency"],
    "polarity": ["anode", "cathode", "stripe"], "long": ["anode"], "short": ["cathode"],
}


def expand_query(tokens: List[str]) -> List[str]:
    out = list(tokens)
    for t in tokens:
        out.extend(SYNONYMS.get(t, []))
    return out


class BM25:
    """Okapi BM25 with an inverted index."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        post: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                post[t].append((i, f))
        self.postings = {t: (np.array([p[0] for p in ps]), np.array([p[1] for p in ps], np.float32))
                         for t, ps in post.items()}
        self.idf = {t: math.log(1 + (self.N - len(ps[0]) + 0.5) / (len(ps[0]) + 0.5))
                    for t, ps in self.postings.items()}
        self._norm = self.k1 * (1 - self.b + self.b * self.dl / max(self.avgdl, 1e-9))

    def scores(self, query: List[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(query):
            p = self.postings.get(t)
            if p is None:
                continue
            ids, tf = p
            s[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + self._norm[ids])
        return s


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._m = None

    @property
    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(self.model_name)
        v = self._m.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


class HybridStore:
    def __init__(self, index_dir: Path, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_dense: bool = True):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[Chunk] = []
        self.emb: Optional[np.ndarray] = None
        self.embedder = Embedder(embed_model)
        self.use_dense = bool(use_dense) and self.embedder.available
        self._bm25: Optional[BM25] = None
        self._faiss = None
        self.load()

    # persistence -----------------------------------------------------------
    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def _emb_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    def load(self) -> None:
        if self._chunks_path.exists():
            self.chunks = [Chunk(**d) for d in json.loads(self._chunks_path.read_text(encoding="utf-8"))]
        if self.use_dense and self._emb_path.exists():
            self.emb = np.load(self._emb_path)
            if len(self.emb) != len(self.chunks):
                log.warning("embedding/chunk count mismatch - re-embedding on next build")
                self.emb = None
        self._rebuild_sparse()

    def save(self) -> None:
        self._chunks_path.write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
                                     encoding="utf-8")
        if self.emb is not None:
            np.save(self._emb_path, self.emb)

    def reset(self) -> None:
        self.chunks, self.emb = [], None
        for p in (self._chunks_path, self._emb_path):
            if p.exists():
                p.unlink()
        self._rebuild_sparse()

    def _rebuild_sparse(self) -> None:
        self._bm25 = BM25([tokenize(c.text + " " + c.source + " " + " ".join(c.tags)) for c in self.chunks]) \
            if self.chunks else None
        self._faiss = None

    # building ------------------------------------------------------------------
    def sources(self) -> List[str]:
        return sorted({c.source for c in self.chunks})

    def add_chunks(self, new: List[Chunk]) -> int:
        known = {c.id for c in self.chunks}
        new = [c for c in new if c.id not in known]
        if not new:
            return 0
        self.chunks.extend(new)
        if self.use_dense:
            try:
                vecs = self.embedder.encode([c.text for c in new])
                self.emb = vecs if self.emb is None else np.vstack([self.emb, vecs])
            except Exception as e:  # model download failed offline etc. -> sparse only
                log.warning("dense embeddings disabled: %s", e)
                self.use_dense, self.emb = False, None
        self._rebuild_sparse()
        self.save()
        return len(new)

    def add_pdf(self, path: Path, size: int = 900, overlap: int = 150) -> int:
        path = Path(path)
        if path.name in self.sources():
            return 0
        return self.add_chunks(pdf_to_chunks(path, size, overlap, load_part_keywords(path.parent)))

    def add_text(self, text: str, source: str, tags=None, page: int = 1) -> int:
        return self.add_chunks(text_to_chunks(text, source, page, tags))

    def build_from_dir(self, pdf_dir: Path, size: int = 900, overlap: int = 150) -> int:
        n = 0
        for p in sorted(Path(pdf_dir).rglob("*.pdf")):
            try:
                added = self.add_pdf(p, size, overlap)
                n += added
                log.info("indexed %s (+%d chunks)", p.name, added)
            except Exception as e:
                log.warning("skip %s: %s", p.name, e)
        return n

    # search --------------------------------------------------------------------
    def _dense_scores(self, query: str) -> Optional[np.ndarray]:
        if not self.use_dense or self.emb is None:
            return None
        q = self.embedder.encode([query])[0]
        try:
            import faiss

            if self._faiss is None:
                self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
                self._faiss.add(self.emb)
            s = np.full(len(self.chunks), -1.0, np.float32)
            D, I = self._faiss.search(q[None], min(50, len(self.chunks)))
            s[I[0]] = D[0]
            return s
        except ImportError:
            return self.emb @ q

    def search(self, query: str, k: int = 4, component: Optional[str] = None, label_boost: float = 0.15,
               part_number: Optional[str] = None) -> List[Citation]:
        if not self.chunks:
            return []
        n = len(self.chunks)
        rrf = np.zeros(n, dtype=np.float32)
        K = 60.0
        sparse = self._bm25.scores(expand_query(tokenize(query))) if self._bm25 else np.zeros(n, np.float32)
        top_sparse = np.argsort(-sparse)[:50]
        for rank, i in enumerate(top_sparse):
            if sparse[i] > 0:
                rrf[i] += 1.0 / (K + rank)
        dense = self._dense_scores(query)
        if dense is not None:
            for rank, i in enumerate(np.argsort(-dense)[:50]):
                rrf[i] += 1.0 / (K + rank)
        if rrf.max() <= 0:
            return []
        rrf = rrf / rrf.max()
        for i, c in enumerate(self.chunks):
            if rrf[i] <= 0:
                continue
            if component and c.topic == component:
                rrf[i] += label_boost          # page is *about* the pointed part
            elif component and component in c.tags:
                rrf[i] += label_boost / 3      # page merely mentions it
            if part_number and part_number.lower() in (c.source.lower() + " " + c.text.lower()):
                rrf[i] += 2 * label_boost
        top = np.argsort(-rrf)[:k]
        return [Citation(self.chunks[i].source, self.chunks[i].page, self.chunks[i].text, float(rrf[i]))
                for i in top if rrf[i] > 0]


def extractive_answer(question: str, cites: List[Citation], max_sentences: int = 3) -> str:
    """No-LLM fallback: the most relevant sentences from the retrieved passages, with citations.

    Sentences are scored by BM25 against the question (IDF computed over the retrieved sentences),
    weighted by the passage's retrieval rank, and returned in document order.
    """
    if not cites:
        return "That isn't in the lab manual or datasheets I have."
    sents: List[Tuple[int, int, str]] = []
    for ci, c in enumerate(cites):
        lines = [ln for ln in c.snippet.splitlines() if ln.strip()]
        # drop a heading line (short, no sentence punctuation) so it doesn't glue onto sentence 1
        if lines and len(lines[0]) < 90 and not re.search(r"[.!?:]\s*$", lines[0]) and len(lines) > 1:
            lines = lines[1:]
        flat = re.sub(r"\s+", " ", " ".join(lines))
        for si, s in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", flat)):
            if len(s) > 12:
                sents.append((ci, si, s.strip()))
    if not sents:
        return f"{cites[0].snippet[:300]} [{cites[0].source} p.{cites[0].page}]"
    bm = BM25([tokenize(s) for _, _, s in sents])
    sc = bm.scores(expand_query(tokenize(question)))
    sc = sc * np.array([1.0 / (1 + 0.3 * ci) for ci, _, _ in sents], np.float32)
    top = float(sc.max())
    best = [i for i in np.argsort(-sc)[:max_sentences] if sc[i] > 0 and sc[i] >= 0.35 * top] or [0]
    best.sort(key=lambda i: (sents[i][0], sents[i][1]))
    parts: List[str] = []
    for j, i in enumerate(best):
        ci, _, s = sents[i]
        end_of_group = j == len(best) - 1 or sents[best[j + 1]][0] != ci
        parts.append(s + (f" [{cites[ci].source} p.{cites[ci].page}]" if end_of_group else ""))
    return " ".join(parts)


def build_store(cfg) -> HybridStore:
    return HybridStore(cfg.path("rag.index_dir", "data/index"), cfg.get("rag.embed_model"),
                       use_dense=cfg.get("rag.dense", True))
