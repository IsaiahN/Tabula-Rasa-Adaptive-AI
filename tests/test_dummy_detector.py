import asyncio
from src.vision.dummy_detector import DummyDetector


async def _run_detect():
    det = DummyDetector()
    frame = [[0]*32 for _ in range(32)]
    objs = await det.detect_objects(frame)
    assert isinstance(objs, list)
    point = await det.detect_actionable_point(frame)
    assert 'x' in point and 'y' in point


def test_dummy_detector_event_loop():
    asyncio.get_event_loop().run_until_complete(_run_detect())
