"""Query composer: fuses what you SAY with what you POINT AT (the core novelty of LabLens).

    speech  : "what is this and where does it connect?"
    pointing: ic (conf 0.91), 8-pin, track #2
    ->  retrieval query : "ic NE555 pinout connections what is this and where does it connect"
    ->  prompt header   : "[POINTED COMPONENT: IC (8-pin), confidence 0.91] [GUIDED STEP 3: ...]"

Also handles follow-ups ("and pin 4?") by carrying the previous component / part number forward.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from lablens.types import Detection

# NE555, LM358N, 74HC595, CD4017, 1N4007, 2N2222, BC547, ATmega328P, L293D ...
PART_RE = re.compile(
    r"\b((?:NE|LM|LF|TL|CD|SN|MC|UA|L|ULN|ATMEGA|ATTINY|BC|BD|TIP|IRF)\s?\d{2,5}[A-Z]{0,3}"
    r"|74(?:HC|HCT|LS|AC)?\d{2,4}|1N\d{3,4}|2N\d{4}|ESP32|ESP8266)\b",
    re.I,
)
PRONOUNS = re.compile(r"\b(this|that|it|its|these|those|here|there|one)\b", re.I)
FOLLOWUP = re.compile(r"^\s*(and|what about|how about|also|then|so)\b", re.I)

# Spoken-number normalisation for STT output ("five five five" -> "555")
_DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
           "six": "6", "seven": "7", "eight": "8", "nine": "9"}

INTENT_HINTS = {
    "pinout": ["pin", "pinout", "connect", "where does", "ground", "gnd", "vcc", "power", "leg"],
    "rating": ["voltage", "current", "rating", "max", "maximum", "power", "watt"],
    "identify": ["what is", "what's this", "identify", "which"],
    "polarity": ["polarity", "positive", "negative", "anode", "cathode", "longer leg", "+", "-"],
}


def normalise_spoken_numbers(text: str) -> str:
    def repl(m):
        return "".join(_DIGITS[w.lower()] for w in m.group(0).split())
    pat = r"\b(?:(?:%s)\s+){1,5}(?:%s)\b" % ("|".join(_DIGITS), "|".join(_DIGITS))
    text = re.sub(pat, repl, text, flags=re.I)
    return re.sub(r"\b(ne|lm|cd)\s+(\d{3})", lambda m: m.group(1).upper() + m.group(2), text, flags=re.I)


def extract_part_number(text: str) -> Optional[str]:
    m = PART_RE.search(normalise_spoken_numbers(text))
    return m.group(1).replace(" ", "").upper() if m else None


def detect_intents(question: str) -> List[str]:
    q = question.lower()
    return [k for k, words in INTENT_HINTS.items() if any(w in q for w in words)]


@dataclass
class Turn:
    question: str
    answer: str
    component: Optional[str]
    part_number: Optional[str]


@dataclass
class QueryContext:
    component: Optional[Detection] = None
    part_number: Optional[str] = None
    guided_step: Optional[str] = None
    extra: List[str] = field(default_factory=list)  # e.g. "resistor value 220 ohm (colour bands)"


@dataclass
class ComposedQuery:
    retrieval: str
    header: str
    component_label: Optional[str]
    part_number: Optional[str]
    intents: List[str]


class Conversation:
    """Interactive RAG memory: keeps the component + step context across follow-up questions."""

    def __init__(self, max_turns: int = 6):
        self.turns: List[Turn] = []
        self.max_turns = max_turns

    def add(self, t: Turn) -> None:
        self.turns.append(t)
        self.turns = self.turns[-self.max_turns:]

    @property
    def last(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    def as_messages(self, n: int = 3) -> List[dict]:
        msgs = []
        for t in self.turns[-n:]:
            msgs += [{"role": "user", "content": t.question}, {"role": "assistant", "content": t.answer}]
        return msgs

    def clear(self) -> None:
        self.turns.clear()


def _is_followup(question: str) -> bool:
    return bool(FOLLOWUP.search(question)) or len(question.split()) <= 4


def compose(question: str, ctx: QueryContext, convo: Optional[Conversation] = None) -> ComposedQuery:
    question = normalise_spoken_numbers(question.strip())
    part = ctx.part_number or extract_part_number(question)
    label = ctx.component.label if ctx.component else None
    last = convo.last if convo else None

    # Follow-up / pronoun resolution: inherit component + part number from the last turn
    if last and (_is_followup(question) or (PRONOUNS.search(question) and not label)):
        label = label or last.component
        part = part or last.part_number

    intents = detect_intents(question)
    expansions = {"pinout": "pinout pin configuration connection diagram",
                  "rating": "absolute maximum ratings electrical characteristics",
                  "polarity": "polarity anode cathode marking",
                  "identify": "description overview features"}
    retrieval = " ".join(x for x in [label or "", part or "", *[expansions[i] for i in intents], question] if x)

    bits = []
    if ctx.component:
        d = ctx.component
        bits.append(f"[POINTED COMPONENT: {d.label.upper()} (detector confidence {d.conf:.2f})]")
    elif label:
        bits.append(f"[PREVIOUSLY DISCUSSED COMPONENT: {label.upper()}]")
    else:
        bits.append("[POINTED COMPONENT: none - the student is not pointing at anything]")
    if part:
        bits.append(f"[PART NUMBER MENTIONED: {part}]")
    if ctx.guided_step:
        bits.append(f"[GUIDED BUILD STEP: {ctx.guided_step}]")
    bits += [f"[{e}]" for e in ctx.extra]
    return ComposedQuery(retrieval, " ".join(bits), label, part, intents)


def cache_key(cq: ComposedQuery, question: str) -> Tuple[str, str]:
    """Normalised key so 'which pin is ground??' and 'Which pin is GND' hit the same cache entry."""
    q = normalise_spoken_numbers(question.lower())
    q = re.sub(r"\bgnd\b", "ground", q)
    q = re.sub(r"\b(the|a|an|is|of|on|this|that|it|please|lablens|hey)\b", " ", q)
    q = " ".join(sorted(set(re.findall(r"[a-z0-9]+", q))))
    return (f"{cq.component_label or '-'}|{cq.part_number or '-'}", q)
