import torch


class _ActorStub:
    def __init__(self, action):
        self._action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)

    def __call__(self, _tensor_dict):
        return {"action": self._action}
