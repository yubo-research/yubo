import time


class ATITimeoutError(Exception):
    pass


class ATIStopped(Exception):
    pass


class AskTellInverter:
    def __init__(self, timeout_seconds=0.1):
        self._x = None
        self._timeout_seconds = timeout_seconds
        self._running = True

    def stop(self):
        self._running = False

    def __call__(self, x):
        self._y = None
        self._x = x
        t_0 = time.time()
        while self._running and self._y is None:
            time.sleep(0.001)
            if time.time() - t_0 > self._timeout_seconds:
                raise ATITimeoutError("No one asked for these arms")
        if not self._running:
            raise ATIStopped
        return self._y

    def ask(self):
        t_0 = time.time()
        while self._running and self._x is None:
            time.sleep(0.001)
            if time.time() - t_0 > self._timeout_seconds:
                raise ATITimeoutError("No arms to ask for")
        if not self._running:
            raise ATIStopped
        return self._x

    def tell(self, y):
        assert len(y) == len(self._x), ("Wrong number of arms", len(y), len(self._x))
        self._x = None
        self._y = y
