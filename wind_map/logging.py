"""
wind_map.logging — File logging of warnings, errors and exit codes.

Every process (train.py, hp_search.py, ...) appends to a single
universal log file in the project root. Entries are timestamped and
carry a per-process run id plus a train/hp label derived from the
invoking script name.

The file only captures WARNING and above (warnings, errors, uncaught
exceptions and exit-code lines). Console output is left untouched.
"""

import atexit
import logging
import os
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOG_FILE = _PROJECT_ROOT / 'run.log'

_configured = False
_exit_code = 0

RUN_ID = uuid.uuid4().hex[:8]
LABEL = (
    os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if sys.argv and sys.argv[0] else 'python')
if LABEL.startswith('hp_'):
    LABEL = 'hp'


class _RunContextFilter(logging.Filter):
    def filter(self, record):
        record.run_id = RUN_ID
        record.label = LABEL
        return True


_FORMAT = (
    '%(asctime)s | run=%(run_id)s | %(label)s | '
    '%(levelname)s | %(message)s')


def _set_exit_code(code):
    """Record the process exit code for the atexit handler."""
    global _exit_code
    _exit_code = code


def _log_uncaught_exception(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        logging.warning('Interrupted (KeyboardInterrupt).')
        _set_exit_code(130)
    else:
        logging.critical(
            'Uncaught exception:', exc_info=(exc_type, exc, tb))
        _set_exit_code(1)


def _log_exit():
    logging.warning(
        f'Process terminated, exit code {_exit_code}.')


def setup_logging(log_file=None):
    """Configure file logging of warnings/errors/exit codes.

    Idempotent: only the first call per process installs handlers,
    so repeated calls (e.g. from every hp_search train() run) simply
    append to the same universal log file.

    Returns the path of the log file.
    """
    global _configured
    if _configured:
        return str(log_file or _DEFAULT_LOG_FILE)
    _configured = True

    if log_file is None:
        log_file = _DEFAULT_LOG_FILE
    log_file = Path(log_file)

    handler = logging.FileHandler(
        log_file, mode='a', encoding='utf-8')
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_RunContextFilter())

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)

    logging.captureWarnings(True)

    sys.excepthook = _log_uncaught_exception
    atexit.register(_log_exit)

    return str(log_file)


def run_script(main_fn):
    """Run *main_fn* and log the resulting exit code.

    The exit-code line itself is written by the atexit handler so it
    is logged exactly once. Codes: 0 on success, 130 on
    KeyboardInterrupt, 1 on any uncaught exception.
    """
    try:
        main_fn()
    except KeyboardInterrupt:
        _set_exit_code(130)
        logging.warning('Interrupted (KeyboardInterrupt).')
        sys.exit(130)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        _set_exit_code(code)
        raise
    except Exception:
        _set_exit_code(1)
        logging.critical('Uncaught exception:', exc_info=True)
        sys.exit(1)
