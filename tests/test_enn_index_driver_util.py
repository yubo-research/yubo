"""Tests for shared ENN index-driver resolution."""

import pytest
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from optimizer.enn_index_driver_util import resolve_enn_index_driver


def test_resolve_enn_index_driver_defaults_and_aliases():
    assert resolve_enn_index_driver(None) is ENNIndexDriver.FLAT
    assert resolve_enn_index_driver("flat") is ENNIndexDriver.FLAT
    assert resolve_enn_index_driver("exact") is ENNIndexDriver.FLAT
    assert resolve_enn_index_driver("bpann_disk") is ENNIndexDriver.BPANN_DISK
    assert resolve_enn_index_driver("bpanndisk") is ENNIndexDriver.BPANN_DISK


def test_resolve_enn_index_driver_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid index_driver"):
        resolve_enn_index_driver("nope")
