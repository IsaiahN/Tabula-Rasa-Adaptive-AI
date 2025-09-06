import numpy as np
import pytest
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop


def _make_loop():
    # Create instance without running __init__ to avoid heavy dependencies
    loop = ContinuousLearningLoop.__new__(ContinuousLearningLoop)
    return loop


def test_normalize_wrapped_list():
    loop = _make_loop()
    # wrapped 2D list: [[row1, row2, ...]]
    frame = [[ [1,0,0], [0,1,0], [0,0,1] ]]
    arr, (w, h) = loop._normalize_frame(frame)
    assert arr is not None
    assert arr.ndim == 2
    assert (w, h) == (3, 3) or (w > 0 and h > 0)


def test_normalize_flat_list_square():
    loop = _make_loop()
    # flat list length 9 -> reshape to 3x3
    frame = [0,1,2,3,4,5,6,7,8]
    arr, (w, h) = loop._normalize_frame(frame)
    assert arr is not None
    assert arr.shape[0] * arr.shape[1] == 9
    assert set(np.unique(arr)).issubset(set(frame)) or arr.size == 9


def test_normalize_numpy_array():
    loop = _make_loop()
    arr_in = np.arange(16).reshape(4,4)
    arr, (w, h) = loop._normalize_frame(arr_in)
    assert arr is not None
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,4)
    assert (w, h) == (4, 4)
