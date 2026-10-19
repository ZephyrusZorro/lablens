"""Material-based circuit synthesizer: designs viable lab projects from available components."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lablens.guided.engine import Project


@dataclass
class SynthesizedProject:
    id: str
    title: str
    difficulty: str  # "Beginner" | "Intermediate" | "Advanced"
    summary: str
    working_principle: str
    formula: str
    required_materials: List[str]
    missing_materials: List[str]
    breadboard_steps: List[str]
    safety_tips: List[str]
    citations: List[str] = field(default_factory=list)
    raw_guided_project: Optional[Dict[str, Any]] = None

    def to_guided_project(self) -> Project:
        """Converts into a camera-verifiable Guided Build Project."""
        if self.raw_guided_project:
            return Project(self.raw_guided_project)
        steps_data = []
        for i, step_text in enumerate(self.breadboard_steps):
            expect = []
            lower = step_text.lower()
            if "breadboard" in lower:
                expect.append({"label": "breadboard", "min": 1, "max": 1})
            if "resistor" in lower:
                expect.append({"label": "resistor", "min": 1})
            if "led" in lower:
                expect.append({"label": "led", "min": 1})
            if "555" in lower or "ic" in lower:
                expect.append({"label": "ic", "min": 1})
            if "capacitor" in lower:
                expect.append({"label": "capacitor", "min": 1})
            if "transistor" in lower:
                expect.append({"label": "transistor", "min": 1})
            if "diode" in lower and "led" not in lower:
                expect.append({"label": "diode", "min": 1})
            if not expect:
                expect.append({"label": "breadboard", "min": 1})
            steps_data.append({
                "instruction": step_text,
                "expect": expect,
                "tip": self.safety_tips[i % len(self.safety_tips)] if self.safety_tips else "Check pins carefully."
            })
        data = {
            "id": self.id,
            "title": self.title,
            "manual_page": "Synthesized Material Project",
            "finish": f"Congratulations! {self.title} is fully assembled. Double check polarity before power.",
            "steps": steps_data
        }
        return Project(data)


# Canonical vocabulary mapping for component keywords
CANONICAL_SYNONYMS = {
    "breadboard": ["breadboard", "board", "protoboard", "bread board"],
    "led": ["led", "light emitting diode", "leds", "indicator"],
    "resistor": ["resistor", "resistors", "ohm", "kohm", "330", "1k", "10k", "220", "470"],
    "capacitor": ["capacitor", "capacitors", "cap", "uf", "electrolytic", "ceramic", "100uf", "10uf", "0.1uf"],
    "ic": ["ic", "555", "ne555", "timer", "lm358", "opamp", "op-amp", "integrated circuit"],
    "transistor": ["transistor", "bc547", "2n2222", "npn", "pnp", "bjt"],
    "diode": ["diode", "1n4007", "rectifier", "1n4148"],
    "zener": ["zener", "zener diode"],
    "power": ["battery", "9v", "5v", "power", "vcc", "usb", "supply"],
    "potentiometer": ["potentiometer", "pot", "variable resistor", "trimmer"],
    "ldr": ["ldr", "light dependent resistor", "photoresistor", "photocell"],
    "switch": ["switch", "button", "pushbutton", "tactile switch"],
}


def normalize_materials(materials: List[str]) -> set[str]:
    """Extracts recognized canonical categories from freeform strings."""
    found = set()
    for item in materials:
        text = item.lower().strip()
        for canonical, aliases in CANONICAL_SYNONYMS.items():
            if any(alias in text for alias in aliases):
                found.add(canonical)
    return found


# Standard structured circuit recipes
CIRCUIT_RECIPES = [
    {
        "id": "recipe_led_indicator",
        "title": "Basic LED Indicator with Current Limiter",
        "difficulty": "Beginner",
        "required": ["breadboard", "led", "resistor"],
        "summary": "The fundamental electronic circuit. Protects a light-emitting diode from overcurrent while demonstrating basic Ohm's Law.",
        "working_principle": "LEDs are current-driven diodes with very low internal resistance once forward-biased ($V_f \\approx 1.8\\text{--}2.2\\text{ V}$). Without a series resistor, excessive current will permanently destroy the semiconductor PN junction.",
        "formula": "R = \\frac{V_{supply} - V_{LED}}{I_{LED}} = \\frac{5\\text{ V} - 2.0\\text{ V}}{10\\text{ mA}} \\approx 300\\text{--}330\\,\\Omega",
        "breadboard_steps": [
            "Place the breadboard firmly on your bench with the center divider horizontal.",
            "Insert the current-limiting resistor (220 to 470 ohm) spanning row 10 and row 15.",
            "Insert the LED: long leg (anode +) to row 15 (connected to resistor), short leg (cathode -) to row 20.",
            "Connect the power supply: positive rail (+5V) to row 10, negative rail (GND) to row 20."
        ],
        "safety_tips": [
            "Always identify the longer lead of the LED as Anode (+) and the flat rim notch as Cathode (-).",
            "Never connect an LED directly across a 5V or 9V battery without a series resistor."
        ],
        "citations": ["LabLens_Lab_Manual.pdf p.3 (LEDs & Polarity)", "LabLens_Lab_Manual.pdf p.2 (Resistors)"]
    },
    {
        "id": "recipe_555_blinker",
        "title": "NE555 Astable LED Pulse Flasher (~1 Hz)",
        "difficulty": "Intermediate",
        "required": ["breadboard", "ic", "resistor", "capacitor", "led"],
        "summary": "Classic astable multivibrator producing continuous digital square waves to blink an LED at approximately 1 pulse per second.",
        "working_principle": "The NE555 timer continuously charges and discharges capacitor C through resistors R1 and R2 between 1/3 Vcc and 2/3 Vcc, driving the output Pin 3 high and low.",
        "formula": "f = \\frac{1.44}{(R_1 + 2R_2) \\cdot C_1}, \\quad T_{high} = 0.693(R_1 + R_2)C_1, \\quad T_{low} = 0.693 R_2 C_1",
        "breadboard_steps": [
            "Place breadboard on bench. Orient the NE555 IC straddling the center trough with notch pointing Left.",
            "Connect Pin 1 to GND (Ground rail) and Pin 8 + Pin 4 (Reset) to +Vcc rail.",
            "Connect Pin 2 (Trigger) directly to Pin 6 (Threshold) with a short jumper wire.",
            "Place Resistor R1 (1k to 10k) between Pin 8 (+Vcc) and Pin 7 (Discharge).",
            "Place Resistor R2 (10k to 100k) between Pin 7 and Pin 6.",
            "Insert electrolytic timing capacitor between Pin 2/6 (+) and Pin 1 (GND).",
            "Connect LED in series with 330 ohm resistor from Output Pin 3 to Ground."
        ],
        "safety_tips": [
            "Observe electrolytic capacitor polarity: the negative stripe must connect to ground Pin 1.",
            "Verify Pin 1 orientation relative to the semicircle notch on the 555 IC."
        ],
        "citations": ["LabLens_Lab_Manual.pdf p.7 (NE555 Timer)", "LabLens_Lab_Manual.pdf p.12 (Astable Circuit)"]
    },
    {
        "id": "recipe_transistor_switch",
        "title": "NPN Transistor (BC547) Low-Power Digital Switch",
        "difficulty": "Beginner",
        "required": ["breadboard", "transistor", "resistor", "led"],
        "summary": "A bipolar junction transistor (BJT) acting as a solid-state switch to control an LED from a delicate low-current input signal.",
        "working_principle": "A small base current ($I_B$) drives the transistor into saturation, allowing a much larger collector current ($I_C = \\beta I_B$) to flow from collector to emitter, illuminating the LED load.",
        "formula": "I_B = \\frac{V_{in} - V_{BE}}{R_B} = \\frac{5\\text{ V} - 0.7\\text{ V}}{R_B}, \\quad I_C \\le \\beta I_B",
        "breadboard_steps": [
            "Place breadboard. Insert BC547 NPN transistor with flat face toward you (Pins: 1=Collector, 2=Base, 3=Emitter).",
            "Connect Pin 3 (Emitter) directly to Ground rail.",
            "Place Base resistor (1k to 10k ohm) connected to Pin 2 (Base).",
            "Connect LED cathode (-) to Pin 1 (Collector) and LED anode (+) in series with 330 ohm resistor to +Vcc.",
            "Apply a control voltage or finger touch to the Base resistor to toggle the LED."
        ],
        "safety_tips": [
            "Do not swap Collector and Emitter leads: check the flat package face orientation.",
            "Always include a base resistor to prevent exceeding maximum base-emitter current."
        ],
        "citations": ["LabLens_Lab_Manual.pdf p.6 (Bipolar Junction Transistors)", "LabLens_Lab_Manual.pdf p.13 (Transistor Switch)"]
    },
    {
        "id": "recipe_diode_clipper",
        "title": "Diode Reverse Polarity Protection & Rectifier",
        "difficulty": "Beginner",
        "required": ["breadboard", "diode", "resistor", "led"],
        "summary": "Demonstrates unidirectional current conduction and reverse voltage blocking using a standard PN junction diode.",
        "working_principle": "Diodes conduct when forward-biased ($V_A > V_K$ by $\\approx 0.7\\text{ V}$ for silicon) and block reverse currents, protecting sensitive electronic circuits from reverse battery insertion.",
        "formula": "I_D = I_S \\left( e^{\\frac{q V_D}{n k T}} - 1 \\right), \\quad V_{drop} \\approx 0.7\\text{ V (Silicon)}",
        "breadboard_steps": [
            "Place breadboard flat on bench.",
            "Insert 1N4007 diode: Silver stripe (Cathode) facing right, black body (Anode) on left.",
            "Connect power +5V rail into the diode Anode (left).",
            "Connect the Cathode (right) into a 330 ohm resistor and LED in series to Ground.",
            "Observe LED glow, then reverse the diode to verify zero current conduction."
        ],
        "safety_tips": [
            "The silver or white printed ring on the diode package marks the Cathode (negative side).",
            "Silicon diodes have a ~0.7V forward voltage drop which reduces output voltage slightly."
        ],
        "citations": ["LabLens_Lab_Manual.pdf p.5 (Diodes & Rectification)"]
    },
    {
        "id": "recipe_rc_integrator",
        "title": "RC Timing & Charge-Discharge Circuit",
        "difficulty": "Intermediate",
        "required": ["breadboard", "capacitor", "resistor", "led"],
        "summary": "Demonstrates capacitor exponential charging and discharging behavior, creating a gradual fade-on / fade-off effect.",
        "working_principle": "The capacitor voltage changes exponentially with time constant $\\tau = RC$. The capacitor absorbs energy until $V_C = V_{supply}$ and slowly releases it when disconnected.",
        "formula": "V_c(t) = V_0 \\left(1 - e^{-t/RC}\\right), \\quad \\tau = R \\cdot C \\text{ (Time constant in seconds)}",
        "breadboard_steps": [
            "Place breadboard on bench.",
            "Connect a large electrolytic capacitor (100 uF to 470 uF) across two rows.",
            "Connect capacitor negative pin (white stripe) to Ground rail.",
            "Place a 10k ohm charging resistor between +5V supply and capacitor positive terminal.",
            "Place an LED with 1k resistor in parallel across the capacitor to observe smooth exponential fading."
        ],
        "safety_tips": [
            "Electrolytic capacitors are polarized. Reverse bias can cause venting or rupture.",
            "Do not short-circuit large charged capacitors directly without a discharge resistor."
        ],
        "citations": ["LabLens_Lab_Manual.pdf p.4 (Capacitors & Time Constants)"]
    }
]


class CircuitSynthesizer:
    """Matches student materials to viable electronics projects and synthesizes guided labs."""

    def __init__(self, recipes: Optional[List[dict]] = None):
        self.recipes = recipes or CIRCUIT_RECIPES

    def match_projects(self, raw_materials: List[str]) -> List[SynthesizedProject]:
        """Matches available raw items against recipes, ranked by feasibility."""
        canonical = normalize_materials(raw_materials)
        results = []

        for r in self.recipes:
            required = set(r["required"])
            matched = required.intersection(canonical)
            missing = required - canonical

            # Calculate feasibility score (1.0 = 100% matched)
            score = len(matched) / max(1, len(required))
            if score >= 0.5 or not missing:
                # Format friendly missing text
                missing_friendly = [f"{m.capitalize()} (missing)" for m in missing]
                p = SynthesizedProject(
                    id=r["id"],
                    title=r["title"],
                    difficulty=r["difficulty"],
                    summary=r["summary"],
                    working_principle=r["working_principle"],
                    formula=r["formula"],
                    required_materials=[m.capitalize() for m in matched],
                    missing_materials=missing_friendly,
                    breadboard_steps=r["breadboard_steps"],
                    safety_tips=r["safety_tips"],
                    citations=r.get("citations", [])
                )
                results.append((score, p))

        # Sort by match score descending
        results.sort(key=lambda item: item[0], reverse=True)
        return [p for _, p in results]

    def synthesize_with_llm(self, raw_materials: List[str], router, store=None) -> SynthesizedProject:
        """Synthesizes an entirely custom project using the LLM and RAG citations."""
        items_str = ", ".join(raw_materials) if raw_materials else "standard electronics kit"
        
        # Ground with relevant passages from manual
        cites_text = ""
        citations = []
        if store:
            hits = store.search(f"circuit projects with {items_str}", top_k=3)
            cites_text = "\n".join(f"Manual p.{c.page}: {c.snippet[:200]}" for c in hits)
            citations = [f"{c.source} p.{c.page}" for c in hits]

        prompt = f"""You are an expert electronics lab professor. The student has the following physical materials on their workbench:
{items_str}

Context from lab manual:
{cites_text}

Design a practical, working electronics circuit they can build right now. Provide:
1. Title: short clear name
2. Difficulty: Beginner, Intermediate, or Advanced
3. Summary: 1-2 sentence overview
4. Principle: How the circuit works
5. Formula: key mathematical formula (e.g. Ohm's Law or timing equation)
6. Step-by-Step Breadboard Instructions: 4-6 clear numbered instructions
7. Safety Tips: 2 essential polarity/voltage safety rules
"""
        messages = [
            {"role": "system", "content": "You are LabLens Circuit Synthesizer. Provide direct, grounded electronics designs."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = router.complete(messages)
            # Parse response into SynthesizedProject
            title = "Custom Synthesized Lab Project"
            difficulty = "Intermediate"
            summary = "Custom circuit synthesized from your available bench materials."
            principle = "Uses your available components in a coordinated feedback/load configuration."
            formula = "V = I \\cdot R, \\quad P = V \\cdot I"
            steps = []
            safety = []

            for line in response.splitlines():
                line = line.strip()
                if line.lower().startswith("title:") or line.lower().startswith("1. title:"):
                    title = re.sub(r"^(?:1\.\s*)?title:\s*", "", line, flags=re.I)
                elif line.lower().startswith("difficulty:"):
                    difficulty = line.split(":", 1)[1].strip()
                elif line.lower().startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.lower().startswith("principle:"):
                    principle = line.split(":", 1)[1].strip()
                elif line.lower().startswith("formula:"):
                    formula = line.split(":", 1)[1].strip()
                elif re.match(r"^\d+\.\s+", line):
                    steps.append(line)
                elif line.startswith("-") or line.startswith("*"):
                    safety.append(line.lstrip("-* "))

            if not steps:
                steps = [
                    "Place breadboard on bench.",
                    f"Insert primary components: {items_str}.",
                    "Wire positive rail to component inputs through appropriate limiting resistors.",
                    "Complete circuit path to ground and verify with multimeter."
                ]
            if not safety:
                safety = ["Check component polarities prior to powering on.", "Ensure supply voltage matches component ratings."]

            return SynthesizedProject(
                id=f"custom_{uuid.uuid4().hex[:6]}",
                title=title,
                difficulty=difficulty,
                summary=summary,
                working_principle=principle,
                formula=formula,
                required_materials=raw_materials,
                missing_materials=[],
                breadboard_steps=steps,
                safety_tips=safety,
                citations=citations
            )
        except Exception:
            # Fallback to top matched recipe
            matched = self.match_projects(raw_materials)
            if matched:
                return matched[0]
            # Default fallback
            return self.match_projects(["breadboard", "resistor", "led"])[0]
