class _PstarDistModal:
    _called = None

    @classmethod
    def hook(cls, called):
        cls._called = called

    def __init__(self, app_name, fn_name, job_fn):
        _ = (app_name, fn_name, job_fn)

    def __call__(self, all_args):
        self.__class__._called["dist"] += len(all_args)
