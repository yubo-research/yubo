"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

import optimizer.optimizer_mo
import optimizer.uhd_enn_fit_helpers
import optimizer.uhd_enn_imputer_predict
import optimizer.uhd_enn_imputer_tell
import optimizer.uhd_loop_support
import problems.dm_control_env_core
import problems.dm_control_pixel_wrapper
import problems.dm_control_spaces
import problems.env_conf_backends
import problems.env_conf_bindings
import problems.env_conf_parse
import problems.env_conf_rl
import problems.env_conf_types
import problems.pixel_policies_encoders
import problems.reactor_policy_params
from tests.kiss_tidy_coverage_c_lib import (
    run_dm_pixel_env_conf,
    run_optimizer_mo_and_fit,
    run_reactor_policy_params,
    run_uhd_enn_mixins,
    run_uhd_loop_support,
)


def test_kiss_tidy_optimizer_mo_and_uhd_fit_helpers(monkeypatch):
    _ = (
        optimizer.optimizer_mo,
        optimizer.uhd_enn_fit_helpers,
    )
    run_optimizer_mo_and_fit(monkeypatch)


def test_kiss_tidy_uhd_enn_predict_tell_mixins(monkeypatch):
    _ = (optimizer.uhd_enn_imputer_predict, optimizer.uhd_enn_imputer_tell)
    run_uhd_enn_mixins(monkeypatch)


def test_kiss_tidy_uhd_loop_support(capsys):
    _ = optimizer.uhd_loop_support
    run_uhd_loop_support(capsys)


def test_kiss_tidy_dm_control_core_pixel_spaces_env_conf(monkeypatch):
    _ = (
        problems.dm_control_env_core,
        problems.dm_control_pixel_wrapper,
        problems.dm_control_spaces,
        problems.env_conf_backends,
        problems.env_conf_bindings,
        problems.env_conf_parse,
        problems.env_conf_rl,
        problems.env_conf_types,
        problems.pixel_policies_encoders,
    )
    run_dm_pixel_env_conf(monkeypatch)


def test_kiss_tidy_reactor_policy_params():
    _ = problems.reactor_policy_params
    run_reactor_policy_params()
