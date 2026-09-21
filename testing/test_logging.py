"""
Unit tests for the package logging setup in ``react_ml_test_public.helpers.logging``.

``helpers.logging`` configures the root logger when it is first imported, and it used to
do that with a bare ``logging.basicConfig(...)`` call. ``basicConfig`` does *nothing at
all* when the root logger already has a handler, and the ML stack configures logging at
import time, before this module is reached: the workflow scripts import ``mlptrain``,
which imports ``autode``, whose ``log.py`` calls ``basicConfig`` at CRITICAL unless
``$AUTODE_LOG_LEVEL`` is set. The level and format declared in ``helpers/logging.py``
were therefore silently ignored, and whichever library configured the root logger first
decided whether anything was printed at all. Where nothing lowered the root level from
CRITICAL - e.g. on a machine without the optional ``coloredlogs`` package, which is what
``mlptrain/log.py`` otherwise uses to pull it down to INFO - every ``logger.info`` /
``logger.warning`` call in this package was dropped without a trace, while the script
itself ran to completion.

Every case here needs its own process: the configuration happens once, on first import,
so a fresh interpreter is the only way to exercise a given "who configured the root
logger first" order.

Run from the repository root:

    pytest testing/test_logging.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

INFO_MARKER = "INFO-MARKER"
WARNING_MARKER = "WARNING-MARKER"
DEBUG_MARKER = "DEBUG-MARKER"

# What import of autode/log/log.py leaves behind when $AUTODE_LOG_LEVEL is unset and
# coloredlogs is not installed: a root handler whose level is CRITICAL. mlptrain (and so
# every workflow script) imports autode before react_ml_test_public.helpers.logging.
AUTODE_STYLE_PRE_CONFIG = (
    "import logging;"
    "logging.basicConfig(level=logging.CRITICAL,"
    " format='%(name)-12s: %(levelname)-8s %(message)s')"
)


def run_in_fresh_process(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter, dropping any inherited ``$MLT_LOG_LEVEL``."""
    child_env = {key: value for key, value in os.environ.items() if key != "MLT_LOG_LEVEL"}
    child_env.update(env or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def log_lines(result: subprocess.CompletedProcess, marker: str) -> list[str]:
    """The stderr lines carrying ``marker``."""
    return [line for line in result.stderr.splitlines() if marker in line]


class TestMessagesAreEmitted:
    """``logger.info`` / ``logger.warning`` must reach the console."""

    def test_messages_appear_with_no_prior_configuration(self):
        code = (
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.info('{INFO_MARKER}');"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        assert INFO_MARKER in result.stderr
        assert WARNING_MARKER in result.stderr

    def test_messages_appear_when_root_was_configured_first(self):
        """Regression test: this is the case that silently dropped every message.

        It mirrors a run of e.g. ``run_mlip_ts_opt.py`` in an environment where the
        optional ``coloredlogs`` package is absent, so autode/mlptrain have already
        pointed the root logger at CRITICAL by the time ``helpers.logging`` is imported.
        """
        code = (
            f"{AUTODE_STYLE_PRE_CONFIG};"
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.info('{INFO_MARKER}');"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        assert INFO_MARKER in result.stderr
        assert WARNING_MARKER in result.stderr

    def test_root_level_is_not_left_at_the_critical_default(self):
        """A CRITICAL root level is what discarded the messages."""
        code = (
            f"{AUTODE_STYLE_PRE_CONFIG};"
            "import logging;"
            "import react_ml_test_public.helpers.logging;"
            "level = logging.getLogger().getEffectiveLevel();"
            "assert level <= logging.INFO, logging.getLevelName(level)"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]

    def test_workflow_import_order_still_logs(self):
        """The workflow scripts import mlptrain before this package's logging module.

        mlptrain brings autode (and, when it is importable, coloredlogs) with it, so in
        this order the root logger is always already configured by the time
        ``helpers.logging`` is reached - whichever of those libraries won decides the
        level and format that used to stick. Both are asserted here so the test does not
        depend on whether ``coloredlogs`` happens to be installed on this machine.
        """
        pytest.importorskip("mlptrain")
        code = (
            "import mlptrain;"
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.info('{INFO_MARKER}');"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        for marker, level in ((INFO_MARKER, "INFO"), (WARNING_MARKER, "WARNING")):
            lines = log_lines(result, marker)
            assert len(lines) == 1, result.stderr[-2000:]
            assert re.fullmatch(
                rf"{level} \d{{2}}:\d{{2}}:\d{{2}}: {marker}", lines[0]
            ), lines[0]


class TestLogFormat:
    """The format declared in ``helpers/logging.py`` must win over other libraries'."""

    def test_line_uses_the_package_format(self):
        code = (
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        lines = log_lines(result, WARNING_MARKER)
        assert len(lines) == 1, result.stderr[-2000:]
        # e.g. 'WARNING 17:54:12: WARNING-MARKER'
        assert re.fullmatch(rf"WARNING \d{{2}}:\d{{2}}:\d{{2}}: {WARNING_MARKER}", lines[0]), lines[0]

    def test_format_survives_an_earlier_configuration(self):
        code = (
            f"{AUTODE_STYLE_PRE_CONFIG};"
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        lines = log_lines(result, WARNING_MARKER)
        assert len(lines) == 1, result.stderr[-2000:]
        assert re.fullmatch(rf"WARNING \d{{2}}:\d{{2}}:\d{{2}}: {WARNING_MARKER}", lines[0]), lines[0]


class TestLogLevelFromEnvironment:
    """``$MLT_LOG_LEVEL`` selects the level, whatever configured logging before."""

    def test_default_level_is_info(self):
        code = (
            "import logging;"
            "import react_ml_test_public.helpers.logging;"
            "print(logging.getLevelName(logging.getLogger().getEffectiveLevel()))"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().splitlines()[-1] == "INFO"

    def test_warning_level_hides_info(self):
        code = (
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.info('{INFO_MARKER}');"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code, {"MLT_LOG_LEVEL": "WARNING"})

        assert result.returncode == 0, result.stderr[-2000:]
        assert INFO_MARKER not in result.stderr
        assert WARNING_MARKER in result.stderr

    def test_debug_level_shows_debug(self):
        code = (
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.debug('{DEBUG_MARKER}')"
        )
        result = run_in_fresh_process(code, {"MLT_LOG_LEVEL": "DEBUG"})

        assert result.returncode == 0, result.stderr[-2000:]
        assert DEBUG_MARKER in result.stderr

    def test_debug_level_hides_debug_at_the_default(self):
        code = (
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.debug('{DEBUG_MARKER}')"
        )
        result = run_in_fresh_process(code)

        assert result.returncode == 0, result.stderr[-2000:]
        assert DEBUG_MARKER not in result.stderr

    @pytest.mark.parametrize("level", ["warning", " Warning ", "info"])
    def test_level_name_is_case_and_whitespace_insensitive(self, level):
        code = (
            "from react_ml_test_public.helpers.logging import logger;"
            f"logger.info('{INFO_MARKER}');"
            f"logger.warning('{WARNING_MARKER}')"
        )
        result = run_in_fresh_process(code, {"MLT_LOG_LEVEL": level})

        assert result.returncode == 0, result.stderr[-2000:]
        expected_info = level.strip().upper() != "WARNING"
        assert (INFO_MARKER in result.stderr) is expected_info
        assert WARNING_MARKER in result.stderr

    @pytest.mark.parametrize("level", ["", "  ", "not-a-level"])
    def test_unparseable_level_falls_back_to_info(self, level):
        """An unusable level must degrade to INFO rather than raise on import."""
        code = (
            "import logging;"
            "import react_ml_test_public.helpers.logging;"
            "print(logging.getLevelName(logging.getLogger().getEffectiveLevel()))"
        )
        result = run_in_fresh_process(code, {"MLT_LOG_LEVEL": level})

        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().splitlines()[-1] == "INFO"
