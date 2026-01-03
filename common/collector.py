from collections import deque


class Collector:
    def __init__(self):
        self._lines = deque()

    def __call__(self, line):
        self._lines.append(line)
        print(line)

    def __iter__(self):
        return iter(self._lines)
