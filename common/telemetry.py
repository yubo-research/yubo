class Telemetry:
    def __init__(self):
        self.reset()

    def reset(self):
        self._dt_fit = None
        self._dt_select = None

    def set_dt_fit(self, dt_fit):
        self._dt_fit = dt_fit

    def set_dt_select(self, dt_select):
        self._dt_select = dt_select

    def format(self):
        dt_fit_str = f"{self._dt_fit:.3f}" if self._dt_fit is not None else "N/A"
        dt_sel_str = f"{self._dt_select:.3f}" if self._dt_select is not None else "N/A"
        return f"dt_fit = {dt_fit_str} dt_sel = {dt_sel_str}"
