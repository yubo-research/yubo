"""Monkey-patches for TorchRL to fix known issues.

Applied at import time so they take effect before any env/collector creation.
"""

from __future__ import annotations


# Patch _StepMDP._grab_and_place: check NO_DEFAULT before using val (fixes
# '_NoDefault' object has no attribute 'empty' when spec has keys that data lacks,
# e.g. pixel envs with PixelsToObservation and step_mdp in ParallelEnv/multiprocessing).
def _patch_step_mdp_grab_and_place():
    from tensordict import LazyStackedTensorDict, is_tensor_collection
    from tensordict.utils import is_non_tensor
    from torchrl.data.tensor_specs import NO_DEFAULT_RL as NO_DEFAULT
    from torchrl.envs.utils import _StepMDP

    _original = _StepMDP._grab_and_place

    @classmethod
    def _grab_and_place_patched(
        cls,
        nested_key_dict: dict,
        data_in,
        data_out,
        _allow_absent_keys: bool,
    ):
        for key, subdict in nested_key_dict.items():
            val = data_in._get_str(key, NO_DEFAULT)
            # Guard: do not use val when it is NO_DEFAULT (key missing from data_in).
            # Original code called val.empty() before this check, causing AttributeError.
            if val is NO_DEFAULT:
                if not _allow_absent_keys:
                    raise KeyError(f"key {key} not found.")
                continue
            if subdict is not None:
                val_out = data_out._get_str(key, None)
                if val_out is None or val_out.batch_size != val.batch_size:
                    val_out = val.empty(batch_size=val.batch_size)
                if isinstance(val, LazyStackedTensorDict):
                    val = LazyStackedTensorDict.lazy_stack(
                        [
                            cls._grab_and_place(
                                subdict,
                                _val,
                                _val_out,
                                _allow_absent_keys=_allow_absent_keys,
                            )
                            for (_val, _val_out) in zip(
                                val.unbind(val.stack_dim),
                                val_out.unbind(val_out.stack_dim),
                            )
                        ],
                        dim=val.stack_dim,
                    )
                else:
                    val = cls._grab_and_place(subdict, val, val_out, _allow_absent_keys=_allow_absent_keys)
            if is_non_tensor(val):
                val = val.clone()
            if is_tensor_collection(val):
                val = val.copy()
            data_out._set_str(key, val, validated=True, inplace=False, non_blocking=False)
        return data_out

    _StepMDP._grab_and_place = _grab_and_place_patched


_patch_step_mdp_grab_and_place()
