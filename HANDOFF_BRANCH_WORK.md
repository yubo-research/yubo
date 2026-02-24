# Branch Handoff: `mehul/rl-etc`

## Scope
Delta from `dsweet/sparse` is centered on:
1. BO trust-region extensions (multi-region + ellipsoidal/metric geometry).
2. RL backend cleanup + layout flattening.

## BO Work (Core)
Primary modules:
- `optimizer/trust_region_config.py`
- `optimizer/trust_region_math.py`
- `optimizer/trust_region_utils.py`
- `optimizer/ellipsoidal_trust_region.py`
- `optimizer/metric_trust_region.py`
- `optimizer/multi_turbo_enn_designer.py`
- `optimizer/multi_turbo_enn_allocation.py`
- `optimizer/multi_turbo_enn_scoring.py`
- `optimizer/multi_turbo_enn_state.py`
- `optimizer/turbo_enn_designer_ext.py`

Intent:
- Support non-box trust-region geometries.
- Support multi-TR allocation/scoring.
- Keep trust-region update behavior configurable.

## RL Layout

- `rl/torchrl/...`
- `rl/pufferlib/...`

Notes:
- `rl/builtins.py` anchors `rl.torchrl` + `rl.pufferlib`
