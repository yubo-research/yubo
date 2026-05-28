from __future__ import annotations

from rl.pufferlib.sac.config import SACConfig
from rl.pufferlib.sac.sac_puffer_engine_impl import config_proxy


def test_kiss_tidy_d_sac_puffer_config_proxy():
    assert config_proxy("SACConfig") is SACConfig
