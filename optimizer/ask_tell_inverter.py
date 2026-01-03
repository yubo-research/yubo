import threading


class ATITimeoutError(Exception):
    pass


class ATIStopped(Exception):
    pass


class AskTellInverter:
    def __init__(self, timeout_seconds=10):
        self._x = None
        self._y = None
        self._timeout_seconds = timeout_seconds
        self._running = True
        self._x_ready = threading.Event()
        self._y_ready = threading.Event()

    def stop(self):
        self._running = False
        self._x_ready.set()
        self._y_ready.set()

    def __call__(self, x):
        self._y = None
        self._y_ready.clear()
        self._x = x
        self._x_ready.set()

        if not self._y_ready.wait(timeout=self._timeout_seconds):
            if not self._running:
                raise ATIStopped
            raise ATITimeoutError("No one asked for these arms")
        if not self._running:
            raise ATIStopped
        return self._y

    def ask(self):
        if not self._x_ready.wait(timeout=self._timeout_seconds):
            if not self._running:
                raise ATIStopped
            raise ATITimeoutError("No arms to ask for")
        if not self._running:
            raise ATIStopped
        return self._x

    def tell(self, y):
        assert len(y) == len(self._x), ("Wrong number of arms", len(y), len(self._x))
        self._x = None
        self._x_ready.clear()
        self._y = y
        self._y_ready.set()
