import numpy as np


class _ALE:
    def __init__(self):
        self._n = 0
        self._lives = 3

    def setInt(self, *_args):
        return None

    def setFloat(self, *_args):
        return None

    def setBool(self, *_args):
        return None

    def loadROM(self, *_args):
        return None

    def getMinimalActionSet(self):
        return [0, 1]

    def getLegalActionSet(self):
        return [0, 1]

    def getScreenGrayscale(self):
        return np.zeros((84, 84), dtype=np.uint8)

    def getScreenRGB(self):
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def act(self, _a):
        self._n += 1
        return 1.0

    def game_over(self):
        return self._n > 3

    def reset_game(self):
        self._n = 0

    def lives(self):
        return self._lives
