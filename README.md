

# Installation

## Installation on MacOS

```bash
./admin/install-macos.sh
```

Apologies for the complexity of installation. We're blending lots of algorithms and test environments, much of which is research-quality code, some of which may be unmaintained.


## Installation on Linux

```bash
./admin/install.sh
```

## Setup
```bash
pre-commit install
```


## Verification

```bash
pre-commit run
pytest -sv tests
```

## Jujutsu checks

After creating a Jujutsu workspace, install the repo-local alias once:

```bash
admin/jj_install_aliases.sh
```

Then run gates through jj:

```bash
jj check              # agent gate (default): quick + kiss + pytest (testmon subset)
jj check quick        # ruff, temp-code scan, missing-files only
jj check publish      # quick + kiss + full pytest (--no-testmon); run before push
```

If your code crashes or hangs, try this [hack](https://discuss.pytorch.org/t/ran-into-this-issue-while-executing/101460):
```
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```
I don't recommend this, however, as it may slow things down.

---

## Examples

From the repository root (with `PYTHONPATH` set to the repo so imports resolve):

```bash
./ops/experiment.py local configs/demo/turbo_enn.toml
./ops/experiment.py local configs/demo/ppo.toml
./ops/exp_uhd.py local configs/demo/mezo.toml
```
