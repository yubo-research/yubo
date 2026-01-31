import os
import tempfile
from pathlib import Path


def test_trace_record_dataclass():
    from analysis.data_io import TraceRecord

    record = TraceRecord(i_iter=0, dt_prop=0.1, dt_eval=0.2, rreturn=1.5)
    assert record.i_iter == 0
    assert record.dt_prop == 0.1
    assert record.dt_eval == 0.2
    assert record.rreturn == 1.5


def test_data_writer_file():
    from analysis.data_io import data_writer

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        temp_path = f.name

    try:
        with data_writer(temp_path) as f:
            f.write("test\n")
        with open(temp_path) as f:
            assert f.read() == "test\n"
    finally:
        os.unlink(temp_path)


def test_data_is_done_false():
    from analysis.data_io import data_is_done

    result = data_is_done("/nonexistent/path/file.txt")
    assert result is False


def test_mark_done():
    from analysis.data_io import mark_done

    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        mark_done(temp_path)
        done_marker = Path(temp_path).with_suffix(".done")
        assert done_marker.exists()
        done_marker.unlink()
    finally:
        os.unlink(temp_path)


def test_write_and_read_config():
    from analysis.data_io import read_config, write_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"key": "value", "num": 42}
        write_config(tmpdir, config)
        loaded = read_config(tmpdir)
        assert loaded == config


def test_write_and_read_trace_jsonl():
    from analysis.data_io import TraceRecord, read_trace_jsonl, write_trace_jsonl

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_fn = os.path.join(tmpdir, "trace.jsonl")
        records = [
            TraceRecord(i_iter=0, dt_prop=0.1, dt_eval=0.2, rreturn=1.5),
            TraceRecord(i_iter=1, dt_prop=0.2, dt_eval=0.3, rreturn=2.0),
        ]
        write_trace_jsonl(trace_fn, records)
        loaded = read_trace_jsonl(trace_fn)
        assert len(loaded) == 2
        assert loaded[0].i_iter == 0
        assert loaded[1].i_iter == 1


def test_write_log_jsonl():
    from analysis.data_io import write_log_jsonl

    with tempfile.TemporaryDirectory() as tmpdir:
        log_fn = os.path.join(tmpdir, "log.jsonl")
        lines = ["line1", "line2", "line3"]
        write_log_jsonl(log_fn, lines)
        with open(log_fn) as f:
            content = f.read()
        assert "line1" in content
        assert "line2" in content
        assert "line3" in content
