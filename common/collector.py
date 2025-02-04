class Collector:
    def __init__(self):
        self._lines = []

    def __call__(self, line):
        self._lines.append(line)
        print(line)

    def __next__(self):
        try:
            return self._lines.pop(0)
        except IndexError:
            raise StopIteration()

    def __iter__(self):
        return self
