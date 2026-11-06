import json
import logging

from kairos.logging import JsonFormatter, RunIdFilter, configure_logging, get_logger, get_run_id


def test_get_logger_prefixes_namespace():
    assert get_logger("data.epoch").name == "kairos.data.epoch"
    assert get_logger("kairos.x").name == "kairos.x"
    assert get_logger().name == "kairos"


def test_run_id_is_stable_and_exported(monkeypatch):
    monkeypatch.delenv("KAIROS_RUN_ID", raising=False)
    rid = get_run_id()
    assert rid == get_run_id()
    import os

    assert os.environ["KAIROS_RUN_ID"] == rid


def test_run_id_from_env(monkeypatch):
    monkeypatch.setenv("KAIROS_RUN_ID", "run-42")
    assert get_run_id() == "run-42"


def test_json_lines_carry_run_id(monkeypatch, capsys):
    monkeypatch.setenv("KAIROS_RUN_ID", "run-json")
    configure_logging(level="INFO", fmt="json", force=True)
    try:
        get_logger("test").info("hello %s", "world")
        line = capsys.readouterr().err.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["run_id"] == "run-json"
        assert payload["msg"] == "hello world"
        assert payload["logger"] == "kairos.test"
    finally:
        configure_logging(force=True)


def test_rich_format_includes_run_id(monkeypatch):
    monkeypatch.setenv("KAIROS_RUN_ID", "run-rich")
    record = logging.LogRecord("kairos.t", logging.INFO, __file__, 1, "msg", None, None)
    RunIdFilter().filter(record)
    assert record.run_id == "run-rich"
    assert json.loads(JsonFormatter().format(record))["run_id"] == "run-rich"
