from .config import SACConfig, TrainResult


def train_sac(config):
    from .trainer import train_sac as _train_sac

    return _train_sac(config)


def register():
    from rl.registry import register_algo

    return register_algo("sac", SACConfig, train_sac)


__all__ = ["SACConfig", "TrainResult", "register", "train_sac"]
