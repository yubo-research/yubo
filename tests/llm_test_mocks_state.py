from __future__ import annotations


class FakeState(dict):
    def __init__(self, data):
        super().__init__(data)

    @property
    def completion(self):
        return self["completion"]

    @completion.setter
    def completion(self, value):
        self["completion"] = value

    def __getitem__(self, key):
        if key in {"prompt", "answer", "info", "example_id"} and "input" in self and key in self["input"]:
            return self["input"][key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class FakeAssistantMessage:
    def __init__(self, content, role="assistant", tool_calls=None):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
