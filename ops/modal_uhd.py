import ops.modal_uhd_runner_impl as modal_uhd_runner_impl


def __getattr__(name: str):
    if name == "modal":
        import modal as modal_mod

        return modal_mod
    if name in ("_Tee", "_ENNFields"):
        return getattr(modal_uhd_runner_impl, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def run(*args, **kwargs):
    return modal_uhd_runner_impl.run(*args, **kwargs)
