class Telemetry:
    """Per-iteration designer telemetry for logging.

    ``dt_rollout`` is wall time spent in environment simulation *inside* the
    designer (e.g. PPO on-policy collection). The Optimizer subtracts it from
    ``proposal_dt`` (``dt_prop``) and adds it to ``eval_dt`` so proposal time
    does not include simulation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._dt_fit = None
        self._dt_select = None
        self._dt_rollout = None

    def set_dt_fit(self, dt_fit):
        self._dt_fit = dt_fit

    def set_dt_select(self, dt_select):
        self._dt_select = dt_select

    def set_dt_rollout(self, dt_rollout: float) -> None:
        self._dt_rollout = float(dt_rollout)

    def set_num_rollout_workers(self, n: int) -> None:
        """Optional hook for designers that report parallel rollout width."""

    def rollout_seconds(self) -> float:
        return 0.0 if self._dt_rollout is None else float(self._dt_rollout)

    def format(self):
        dt_fit_str = f"{self._dt_fit:.3f}" if self._dt_fit is not None else "N/A"
        dt_sel_str = f"{self._dt_select:.3f}" if self._dt_select is not None else "N/A"
        return f"fit_dt = {dt_fit_str} select_dt = {dt_sel_str}"
