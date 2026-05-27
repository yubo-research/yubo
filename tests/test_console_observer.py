import asyncio
import io
import json

from llm.console_observer import SplitConsoleObserver, UnifiedConsoleManager


def test_console_observer_routes_and_saves_split_logs(tmp_path):
    stream = io.StringIO()
    observer = SplitConsoleObserver(stream=stream, log_dir=tmp_path, enable_tui=False)

    with observer:
        observer.route_line("EVAL: step = 0 mu = 0.1000")
        observer.route_line("Model: theorem test : True := by trivial")
        observer.route_line("Tool [lean4]: ok")
        observer.route_line("Downloading shard 1/3")

    assert "train | EVAL: step = 0 mu = 0.1000" in stream.getvalue()
    assert "model | Model: theorem test : True := by trivial" in stream.getvalue()
    session_dir = observer.session_log_dir
    assert session_dir is not None
    assert session_dir.parent == tmp_path / "console"
    assert "EVAL: step = 0" in (session_dir / "train.log").read_text(encoding="utf-8")
    model_log = (session_dir / "inference.log").read_text(encoding="utf-8")
    assert "Model: theorem test" in model_log
    assert "Tool [lean4]: ok" in model_log
    assert "Downloading shard" in (session_dir / "diagnostics.log").read_text(encoding="utf-8")
    assert "train       EVAL: step = 0" in (session_dir / "combined.log").read_text(encoding="utf-8")

    events = [json.loads(line) for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["channel"] for event in events] == [
        "train",
        "inference",
        "inference",
        "diagnostics",
    ]


def test_unified_console_manager_attaches_active_observer(tmp_path):
    stream = io.StringIO()
    observer = SplitConsoleObserver(stream=stream, log_dir=tmp_path, enable_tui=False)

    async def run():
        with observer:
            manager = UnifiedConsoleManager()
            assert manager.observers == [observer]
            await manager.broadcast_step(0, {"role": "assistant", "content": "proof attempt"})
            await manager.broadcast_reward(1.0, {"status": "success"})

    asyncio.run(run())

    session_dir = observer.session_log_dir
    assert session_dir is not None
    model_log = (session_dir / "inference.log").read_text(encoding="utf-8")
    assert "[turn 0] assistant" in model_log
    assert "proof attempt" in model_log
    assert "REWARD: 1.0000 status=success" in model_log


def test_console_observer_has_scrollable_viewport(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llm.console_observer.shutil.get_terminal_size",
        lambda fallback: type("Size", (), {"columns": 120, "lines": 12})(),
    )
    observer = SplitConsoleObserver(stream=io.StringIO(), log_dir=tmp_path, enable_tui=False)

    with observer:
        for idx in range(30):
            observer.append_train(f"line {idx}")
        rendered = observer._render_channel("train")
        assert "line 0" not in rendered
        assert "line 29" in rendered

        observer._scroll_active(-4)
        observer.append_train("line 30")
        assert "frozen +1" in observer._pane_title("train")
        observer._follow_active()
        assert "tail" in observer._pane_title("train")


def test_console_observer_does_not_own_terminal_input(tmp_path):
    observer = SplitConsoleObserver(stream=io.StringIO(), log_dir=tmp_path)

    with observer:
        assert observer.owns_terminal_input is False
