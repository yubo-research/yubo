import numpy as np
from ax.service.ax_client import AxClient


class AxDesigner:
    def __init__(self, policy):
        self._policy = policy
        self._ax_client = None
        self._ax_client = AxClient()
        num_params = policy.num_params()
        self._ax_client.create_experiment(
            name=f"ax_{num_params}",
            parameters=[
                {
                    "name": f"x{i:03d}",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                    "value_type": "float",
                }
                for i in range(num_params)
            ],
        )
        self._trial_index = None

    def __call__(self, data):
        import warnings

        warnings.simplefilter("ignore")
        with warnings.catch_warnings():
            if self._trial_index is not None:
                self._ax_client.complete_trial(trial_index=self._trial_index, raw_data=data[-1].trajectory.rreturn)
            parameters, self._trial_index = self._ax_client.get_next_trial()
            p = np.array([parameters.get(f"x{i:03d}") for i in range(self._policy.num_params())])
            policy = self._policy.clone()
            policy.set_params(p)
            return policy
