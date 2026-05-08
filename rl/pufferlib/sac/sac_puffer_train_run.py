from __future__ import annotations


def train_sac_puffer_impl(config):
    from common.im import im

    return im("rl.pufferlib.sac.sac_puffer_train_run_impl").train_sac_puffer_impl(config)


def train_sac_puffer(config):
    return train_sac_puffer_impl(config)
