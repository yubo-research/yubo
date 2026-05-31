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
    def __init__(
        self,
        content,
        role="assistant",
        tool_calls=None,
        reasoning_content=None,
        finish_reason=None,
        is_truncated=None,
        tool_call_id=None,
    ):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.finish_reason = finish_reason
        self.is_truncated = is_truncated
        self.tool_call_id = tool_call_id
