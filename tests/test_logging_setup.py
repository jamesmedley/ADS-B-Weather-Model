"""
wind_map.logging tests: idempotent setup, warning capture, and the
run_script exit-code contract.

The module-level `_configured` flag is process-global, so these tests
reset it around each case and point handlers at tmp_path files to
avoid touching the real run.log in the repo root.
"""

import logging

import pytest

import wind_map.logging as wm_logging
from wind_map.logging import run_script, setup_logging

pytestmark = pytest.mark.unit


@pytest.fixture()
def fresh_logging(tmp_path, monkeypatch):
    log_path = tmp_path / "test_run.log"
    monkeypatch.setattr(wm_logging, "_configured", False)

    yield log_path

    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and \
                getattr(h, "_wm_test", False):
            root.removeHandler(h)
            h.close()


def _tag_handler(log_path):
    """Attach a marker attr to the handler we create."""
    root = logging.getLogger()
    for h in root.handlers:
        base = getattr(h, "baseFilename", "")
        if base == str(log_path):
            h._wm_test = True


def test_setup_logging_returns_path_and_captures_warnings(
        fresh_logging):
    log_path = fresh_logging
    returned = setup_logging(str(log_path))
    _tag_handler(log_path)
    assert returned == str(log_path)
    logging.warning("hello-warning")
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    assert "hello-warning" in content


def test_setup_logging_is_idempotent(fresh_logging):
    log_path = fresh_logging
    setup_logging(str(log_path))
    n_handlers = len(logging.getLogger().handlers)
    returned = setup_logging(str(log_path / "other.log"))
    # Second call is a no-op: returns the requested path but does
    # not install another handler.
    assert len(logging.getLogger().handlers) == n_handlers
    assert not (log_path.parent / "other.log").exists()
    assert returned.endswith("other.log")


def test_run_script_success_logs_exit_code_zero(fresh_logging):
    log_path = fresh_logging
    setup_logging(str(log_path))
    _tag_handler(log_path)
    ran = []

    def main():
        ran.append(True)

    run_script(main)
    assert ran == [True]
    wm_logging._log_exit()
    for h in logging.getLogger().handlers:
        h.flush()
    assert "exit code 0" in log_path.read_text(
        encoding="utf-8")


def test_run_script_swallows_exception_and_exits_1(
        fresh_logging):
    log_path = fresh_logging
    setup_logging(str(log_path))
    _tag_handler(log_path)

    def boom():
        raise RuntimeError("kaput")

    with pytest.raises(SystemExit) as exc:
        run_script(boom)
    assert exc.value.code == 1
    assert wm_logging._exit_code == 1
    for h in logging.getLogger().handlers:
        h.flush()
    assert "kaput" in log_path.read_text(encoding="utf-8")


def test_run_script_keyboard_interrupt_maps_to_130(
        fresh_logging):
    log_path = fresh_logging
    setup_logging(str(log_path))
    _tag_handler(log_path)

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as exc:
        run_script(interrupted)
    assert exc.value.code == 130
