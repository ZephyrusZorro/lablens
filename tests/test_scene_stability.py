"""Regression cases for crowded scenes and detector failures (no hardware required)."""
import time
import numpy as np
from types import SimpleNamespace
from lablens.types import Detection
from lablens.vision.detector import ComponentDetector, AsyncDetector, BaseDetector
from lablens.vision.humans import boxes_match
from lablens.runtime import gesture_controls_allowed


def test_adjacent_people_are_not_deduplicated():
    a = Detection("person", .9, 0, 0, 100, 200)
    b = Detection("person", .9, 80, 0, 180, 200)
    assert not boxes_match(a, b)
    assert boxes_match(a, Detection("head", .9, 10, 10, 50, 60))


def test_multiple_people_have_independent_tracks_and_no_lingering_anatomy():
    detector = ComponentDetector(SimpleNamespace(classes=[]))
    raw = [Detection("person", .9, x, 0, x+50, 100) for x in (0, 100, 200)]
    raw.append(Detection("left arm", .9, 0, 0, 20, 30))
    detector.detect_raw = lambda frame: raw
    frame = np.zeros((100, 300, 3), np.uint8)
    detector.detect(frame)
    result = detector.detect(frame)
    ids = {d.track_id for d in result if d.label == "person"}
    assert len(ids) == 3
    assert ids == {d.track_id for d in detector.detect(frame) if d.label == "person"}
    raw.clear()
    assert not any(d.label == "left arm" for d in detector.detect(frame))
    assert detector.detect(frame) == []


def test_crowd_cannot_advance_build_even_in_mixed_mode():
    people = [Detection("person", .9, x, 0, x+50, 100) for x in (0, 100, 200)]
    board = Detection("breadboard", .9, 0, 200, 300, 300)
    assert not gesture_controls_allowed(people + [board], "mixed")
    assert not gesture_controls_allowed(people + [board], "auto")


def test_failed_detection_clears_previous_boxes_and_recovers():
    class Flaky(BaseDetector):
        fail = False
        def detect(self, frame):
            if self.fail:
                raise ValueError("test failure")
            return [Detection("led", .9, 0, 0, 10, 10)]
    model = Flaky()
    worker = AsyncDetector(model)
    def infer():
        previous = worker.sequence
        worker.submit(np.zeros((10, 10, 3), np.uint8))
        deadline = time.monotonic()+2
        while worker.sequence == previous and time.monotonic() < deadline:
            time.sleep(.01)
        assert worker.sequence != previous
    try:
        infer()
        assert worker.latest()
        model.fail = True
        infer()
        assert not worker.latest() and "test failure" in worker.error
        model.fail = False
        infer()
        assert worker.latest() and not worker.error
        worker.updated_at = time.monotonic()-3
        assert not worker.latest()
    finally:
        worker.close()


def test_manual_selection_is_immediate_and_cannot_jump_to_another_part():
    from lablens.runtime import Runtime, Snapshot
    rt = Runtime.__new__(Runtime)
    rt._snap = Snapshot(detections=[Detection("led", .9, 0, 0, 20, 20, 7)])
    rt.select_component(7)
    assert rt._current_selection().label == "led"
    rt._snap.detections = [Detection("resistor", .9, 0, 0, 20, 20, 8)]
    assert rt._current_selection() is None


def test_missing_models_never_silently_present_fake_detections():
    import pytest
    from lablens.config import Config
    from lablens.vision.detector import build_detector
    with pytest.raises(RuntimeError, match="models missing"):
        build_detector(Config({"vision": {"models": [], "mock": False}}))
