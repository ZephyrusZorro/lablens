"""Prompt templates."""
from __future__ import annotations

from typing import List

from lablens.types import Citation

LANG_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

SYSTEM = """You are LabLens, a patient electronics lab mentor speaking out loud to a student at a workbench.
Rules:
- Answer in {language}. Keep it short enough to speak: 2-4 sentences unless the student asks for detail.
- Ground every factual claim (pin numbers, voltages, ratings) in the CONTEXT. Cite inline as [source p.N].
- If the CONTEXT does not contain the answer, say "That isn't in the datasheets I have" and give only
  general, clearly-labelled advice. Never invent pin numbers.
- The student is pointing at a physical part; "this" / "it" refers to the POINTED COMPONENT.
- The camera only knows the part's type (e.g. "IC"), not its part number. If the CONTEXT covers several
  different parts of that type and no part number was mentioned, ask which number is printed on it,
  or answer briefly for each and say which is which. Never mix pinouts of different parts.
- Safety first: mention polarity, current limiting or power-off when relevant.
- You only see which parts are present on the bench, not the wiring. Don't claim to verify connections."""


def system_prompt(language: str = "en") -> str:
    return SYSTEM.format(language=LANG_NAMES.get(language, language))


def format_context(cites: List[Citation], max_chars: int = 900) -> str:
    if not cites:
        return "(no matching datasheet passages)"
    return "\n\n".join(f"[{c.source} p.{c.page}]\n{c.snippet[:max_chars]}" for c in cites)


def user_prompt(header: str, question: str, cites: List[Citation]) -> str:
    return f"{header}\n\nCONTEXT:\n{format_context(cites)}\n\nSTUDENT QUESTION: {question}"


NO_RAG_SYSTEM = "You are an electronics lab mentor. Answer briefly."
