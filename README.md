
# Installation

## macOS (Apple Silicon)

Use Pixi — see **[admin/README-mac-emps.md](admin/README-mac-emps.md)**.

```bash
pixi install -e hyperscalees
pixi run -e hyperscalees bootstrap-mac   # extras + verify (first time ~1–2 min)
pixi run -e hyperscalees pytest -sv tests -rs
```

Isaac Sim is Modal/Linux only (`pixi install -e isaaclab` on GPU machines).

## Linux

Modal / GPU: see **[admin/README-hyperscalees.md](admin/README-hyperscalees.md)**.

Bare-metal:

```bash
bash admin/setup-hyperscalees.sh
```

## Setup

```bash
pre-commit install
```

## Verification

```bash
pre-commit run
pixi run -e hyperscalees pytest -sv tests -rs   # macOS
# or: pytest -sv tests                          # after setup-hyperscalees.sh on Linux
```

On macOS, `KMP_DUPLICATE_LIB_OK` and faiss import order are handled via Pixi activation and `sitecustomize.py`.

---

## Examples

From the repository root (Pixi sets `PYTHONPATH` on Mac; elsewhere use `PYTHONPATH=.`):

```bash
pixi run -e hyperscalees python ops/experiment.py local configs/demo/turbo_enn.toml
pixi run -e hyperscalees python ops/experiment.py local configs/demo/ppo.toml
pixi run -e hyperscalees python ops/exp_uhd.py local configs/demo/mezo.toml
```
