_Serial = type("Serial", (), {})
_Multiprocessing = type("Multiprocessing", (), {})


class _FakeVector:
    Serial = _Serial
    Multiprocessing = _Multiprocessing

    def __init__(self):
        self.calls = []

    def make(self, env_creator, *, env_kwargs, backend, num_envs, seed, **kwargs):
        self.calls.append(
            {
                "env_creator": env_creator,
                "env_kwargs": env_kwargs,
                "backend": backend,
                "num_envs": num_envs,
                "seed": seed,
                "kwargs": kwargs,
            }
        )
        return "vec-env"


class _FakeAtari:
    def __init__(self):
        self.games = []

    def env_creator(self, game_name):
        self.games.append(game_name)

        def _creator():
            return game_name

        return _creator
