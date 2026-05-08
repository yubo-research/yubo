class _DistCall:
    def get(self, timeout):
        assert timeout == 5
        return {"ok": True}


class _DistFactory:
    @staticmethod
    def from_id(_call_id):
        return _DistCall()
