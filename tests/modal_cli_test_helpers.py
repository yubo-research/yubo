from __future__ import annotations


class OkModalResult:
    returncode = 0


def capture_subprocess_run(module, monkeypatch):
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, env=None, **_kwargs):
        e = {k: v for k, v in (env or {}).items() if isinstance(v, str)}
        calls.append((list(cmd), e))
        return OkModalResult()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


def assert_modal_command_calls(captured, expected_modal_cmds, *, modal_tag: str) -> None:
    assert len(captured) == len(expected_modal_cmds)
    for i, exp_cmd in enumerate(expected_modal_cmds):
        cmd, env = captured[i]
        assert cmd == exp_cmd
        if exp_cmd[1] == "deploy" or (len(exp_cmd) > 2 and "batches" in exp_cmd[2]):
            assert env.get("MODAL_TAG") == modal_tag
