class _Optimizer:
    def __init__(self, func):
        self._func = func

    def optimize(self, num_iterations):
        import numpy as np

        y_max = -1e99
        x_0 = 0.5
        for _ in range(num_iterations):
            x = x_0 + 0.03 * np.random.normal()
            x = min(1, max(0, x))
            y = float(self._func(np.array([x])))
            if y > y_max:
                y_max = y
                x_0 = x


def test_ask_tell_inverter():
    import threading

    from optimizer.ask_tell_inverter import AskTellInverter, ATITimeoutError

    def f(x):
        return -((x - 0.3) ** 2)

    print()

    ati = AskTellInverter()

    def _run_opt():
        opt = _Optimizer(ati)
        try:
            return opt.optimize(100)
        except ATITimeoutError as e:
            print("A:", e)

    thread = threading.Thread(target=_run_opt, args=())
    thread.start()

    y_max = -1e99
    x_max = None
    for _ in range(101):
        try:
            x = ati.ask()
        except ATITimeoutError as e:
            print("B:", e)
            break
        y = f(x)
        if y > y_max:
            y_max = y
            x_max = x
        print("X:", y_max, x_max)
        ati.tell(y)

    thread.join()
