import numpy as np

import common.all_bounds as all_bounds


class AxDesigner:
    def __init__(self, policy):
<<<<<<< HEAD
<<<<<<< HEAD
        from ax.service.ax_client import AxClient

        self._policy = policy
        self._ax_client = AxClient(verbose_logging=False)
        num_params = policy.num_params()
=======
=======
>>>>>>> main
        self._policy = policy
        self._ax_client = None

    def _lazy_init(self):
        if self._ax_client:
            return
        from ax.service.ax_client import AxClient

        self._ax_client = AxClient(verbose_logging=False)
        num_params = self._policy.num_params()
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> main
        self._ax_client.create_experiment(
            name=f"ax_{num_params}",
            parameters=[
                {
                    "name": f"x{i:03d}",
                    "type": "range",
                    "bounds": [all_bounds.x_low, all_bounds.x_high],
                    "value_type": "float",
                }
                for i in range(num_params)
            ],
        )
        self._trial_index = None

<<<<<<< HEAD
<<<<<<< HEAD
    def __call__(self, data):
=======
=======
>>>>>>> main
    def __call__(self, data, num_arms):
        assert num_arms == 1, "ax only supports one-arm trials"
        self._lazy_init()

<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> main
        import warnings

        warnings.simplefilter("ignore")
        with warnings.catch_warnings():
            if self._trial_index is not None:
                self._ax_client.complete_trial(trial_index=self._trial_index, raw_data=data[-1].trajectory.rreturn)
            parameters, self._trial_index = self._ax_client.get_next_trial()
            p = np.array([parameters.get(f"x{i:03d}") for i in range(self._policy.num_params())])
            policy = self._policy.clone()
            policy.set_params(p)
<<<<<<< HEAD
<<<<<<< HEAD
            return policy
=======

        return [policy]
>>>>>>> main
=======

        return [policy]
>>>>>>> main
