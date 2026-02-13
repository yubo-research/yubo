

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

If your code crashes or hangs, try this [hack](https://discuss.pytorch.org/t/ran-into-this-issue-while-executing/101460):
```
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```
I don't recommend this, however, as it may slow things down.
