def test_designer_protocol_exists():
    from optimizer.designer_protocol import Designer

    assert Designer is not None


def test_designer_protocol_is_protocol():
    from typing import Protocol

    from optimizer.designer_protocol import Designer

    assert issubclass(Designer, Protocol)
