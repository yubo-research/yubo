def test_designer_protocol_exists():
    from optimizer.designer_protocol import Designer

    assert Designer is not None


def test_designer_protocol_is_protocol():
    from typing import Protocol

    from optimizer.designer_protocol import Designer

    assert issubclass(Designer, Protocol)


def test_get_designer_algo_metrics():
    from optimizer.designer_protocol import get_designer_algo_metrics

    assert get_designer_algo_metrics(object()) == {}
    assert get_designer_algo_metrics(type("X", (), {})()) == {}

    class WithMetrics:
        def get_algo_metrics(self):
            return {"sigma": 0.5, "iter": 10.0}

    assert get_designer_algo_metrics(WithMetrics()) == {"sigma": 0.5, "iter": 10.0}

    class EmptyMetrics:
        def get_algo_metrics(self):
            return {}

    assert get_designer_algo_metrics(EmptyMetrics()) == {}

    class Raises:
        def get_algo_metrics(self):
            raise RuntimeError("oops")

    assert get_designer_algo_metrics(Raises()) == {}
