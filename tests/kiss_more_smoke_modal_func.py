class ModalBatchesSpawnFunc:
    def __init__(self, spawned):
        self._spawned = spawned

    def spawn_map(self, todo):
        self._spawned["map"].append(list(todo))

    def spawn(self, *payload):
        self._spawned["spawn"].append(payload)
