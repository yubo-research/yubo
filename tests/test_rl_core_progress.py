import math

from rl.core import progress


def test_is_due():
    assert progress.is_due(6, 3)
    assert not progress.is_due(5, 3)
    assert not progress.is_due(5, 0)
    assert not progress.is_due(5, None)


def test_due_mark():
    assert progress.due_mark(0, 10, 0) is None
    assert progress.due_mark(10, 10, 0) == 1
    assert progress.due_mark(29, 10, 2) is None
    assert progress.due_mark(30, 10, 2) == 3
    assert progress.due_mark(30, None, 2) is None


def test_steps_per_second():
    assert progress.steps_per_second(100, started_at=10.0, now=12.0) == 50.0
    assert math.isnan(progress.steps_per_second(100, started_at=10.0, now=10.0))
