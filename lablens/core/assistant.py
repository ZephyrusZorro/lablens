"""The LabLens brain: (speech + pointing) -> retrieval -> LLM router -> streamed, cited answer."""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, List, Optional

from lablens.core.query import ComposedQuery, Conversation, QueryContext, Turn, cache_key, compose
from lablens.llm.prompts import NO_RAG_SYSTEM, system_prompt, user_prompt
from lablens.llm.router import BackendError
from lablens.rag.store import extractive_answer
from lablens.types import Answer, Citation, Detection

log = logging.getLogger(__name__)


class StreamedAnswer:
    """Iterate to get tokens (feed straight into TTS); .answer is filled when the stream ends."""

    def __init__(self, citations: List[Citation], composed: ComposedQuery, tokens: Iterator[str],
                 on_done: Callable[[str], Answer], cached: Optional[Answer] = None):
        self.citations, self.composed = citations, composed
        self._tokens, self._on_done = tokens, on_done
        self.answer: Optional[Answer] = cached
        self.text = cached.text if cached else ""

    def __iter__(self) -> Iterator[str]:
        if self.answer is not None and self.answer.cached:
            yield self.answer.text
            return
        buf = []
        for t in self._tokens:
            buf.append(t)
            yield t
        self.text = "".join(buf)
        self.answer = self._on_done(self.text)


class LabLens:
    def __init__(self, cfg, store=None, router=None, cache=None, events=None, scraper=None):
        from lablens.llm.router import LLMRouter
        from lablens.rag.store import build_store

        self.cfg = cfg
        self.store = store if store is not None else build_store(cfg)
        self.router = router if router is not None else LLMRouter.from_config(cfg)
        self.cache = cache
        self.events = events
        self.scraper = scraper
        self.convo = Conversation()
        self.language = cfg.get("audio.language", "en")
        self.top_k = cfg.get("rag.top_k", 4)
        self.label_boost = cfg.get("rag.label_boost", 0.15)
        self.notices: List[str] = []  # UI-visible side events ("downloaded NE555 datasheet")

    @classmethod
    def from_config(cls, cfg) -> "LabLens":
        from lablens.core.cache import AnswerCache
        from lablens.dashboard.events import EventLog
        from lablens.rag.scraper import DatasheetScraper

        cache = AnswerCache(cfg.path("cache.path", "data/answer_cache.sqlite")) if cfg.get("cache.enabled", True) else None
        scraper = DatasheetScraper.from_config(cfg) if cfg.get("scraper.enabled", True) else None
        return cls(cfg, cache=cache, events=EventLog(cfg.path("dashboard.db_path", "data/lablens_events.sqlite")),
                   scraper=scraper)

    # ------------------------------------------------------------------------
    def _maybe_scrape(self, part: Optional[str]) -> None:
        if not part or self.scraper is None:
            return
        if any(part.lower() in s.lower() for s in self.store.sources()):
            return
        try:
            path = self.scraper.fetch(part)
        except Exception as e:
            log.warning("scrape %s failed: %s", part, e)
            path = None
        if path:
            n = self.store.add_pdf(path)
            self.notices.append(f"Fetched datasheet for {part} ({path.name}, {n} chunks indexed)")
            if self.events:
                self.events.log("scraped", component=part, file=path.name, chunks=n)
        else:
            self.notices.append(f"Couldn't find a datasheet for {part} online")

    def ask_stream(self, question: str, pointed: Optional[Detection] = None, guided_step: Optional[str] = None,
                   extra: Optional[List[str]] = None, project: str = "", step: int = -1) -> StreamedAnswer:
        t0 = time.perf_counter()
        cq = compose(question, QueryContext(pointed, None, guided_step, extra or []), self.convo)
        key = cache_key(cq, question)
        if self.cache and not guided_step:
            hit = self.cache.get(key)
            if hit:
                hit.latency_s = time.perf_counter() - t0
                self.convo.add(Turn(question, hit.text, cq.component_label, cq.part_number))
                if self.events:
                    self.events.log("question", project, step, cq.component_label or "", q=question,
                                    cached=True, latency=hit.latency_s)
                return StreamedAnswer(hit.citations, cq, iter(()), lambda t: hit, cached=hit)

        self._maybe_scrape(cq.part_number)
        cites = self.store.search(cq.retrieval, self.top_k, cq.component_label, self.label_boost, cq.part_number)
        messages = [{"role": "system", "content": system_prompt(self.language)}]
        messages += self.convo.as_messages()
        messages.append({"role": "user", "content": user_prompt(cq.header, question, cites)})

        def done(text: str) -> Answer:
            ans = Answer(text, cites, self.router.last_backend, time.perf_counter() - t0)
            self.convo.add(Turn(question, text, cq.component_label, cq.part_number))
            if self.cache and not guided_step:
                self.cache.put(key, ans)
            if self.events:
                self.events.log("question", project, step, cq.component_label or "", q=question,
                                backend=ans.backend, latency=round(ans.latency_s, 3),
                                ttft=round(self.router.last_ttft, 3), n_cites=len(cites))
            return ans

        def tokens() -> Iterator[str]:
            try:
                yield from self.router.stream(messages)
            except BackendError as e:
                # No Groq key / no internet / no Ollama: answer straight from the retrieved pages.
                log.info("LLM unavailable (%s) - extractive answer", e)
                self.router.last_backend = "offline-extractive"
                q = question + " " + (cq.component_label or "") + " " + (cq.part_number or "")
                yield extractive_answer(q, cites)

        return StreamedAnswer(cites, cq, tokens(), done)

    def ask(self, question: str, **kw) -> Answer:
        s = self.ask_stream(question, **kw)
        for _ in s:
            pass
        return s.answer

    def ask_without_rag(self, question: str) -> str:
        """Baseline for the grounding evaluation."""
        return self.router.complete([{"role": "system", "content": NO_RAG_SYSTEM},
                                     {"role": "user", "content": question}])

    def pop_notices(self) -> List[str]:
        n, self.notices = self.notices, []
        return n
