"""Unit tests for the material-based circuit synthesizer."""
import pytest
from lablens.core.synthesizer import (
    CircuitSynthesizer,
    normalize_materials,
    SynthesizedProject,
    CIRCUIT_RECIPES
)


def test_normalize_materials():
    raw = ["NE555 Timer IC", "330 ohm resistor", "Red LED", "100uF electrolytic capacitor", "Breadboard"]
    canonical = normalize_materials(raw)
    assert "ic" in canonical
    assert "resistor" in canonical
    assert "led" in canonical
    assert "capacitor" in canonical
    assert "breadboard" in canonical


def test_synthesizer_matches_555_blinker():
    synth = CircuitSynthesizer()
    materials = ["555 timer", "resistors", "capacitors", "red led", "breadboard", "9v battery"]
    matched = synth.match_projects(materials)
    assert len(matched) > 0
    # The 555 astable flasher should be among top matches with high score
    titles = [p.title for p in matched]
    assert any("555" in t for t in titles)


def test_synthesizer_missing_materials_flagged():
    synth = CircuitSynthesizer()
    materials = ["led", "resistor"]  # missing breadboard
    matched = synth.match_projects(materials)
    led_proj = next(p for p in matched if "LED Indicator" in p.title)
    assert any("Breadboard" in m for m in led_proj.missing_materials)


def test_synthesized_project_converts_to_guided_project():
    synth = CircuitSynthesizer()
    materials = ["breadboard", "resistor", "led"]
    matched = synth.match_projects(materials)
    proj = matched[0]
    guided = proj.to_guided_project()
    assert guided.id == proj.id
    assert guided.title == proj.title
    assert len(guided.steps) > 0
    assert any(step["expect"] for step in guided.steps)
