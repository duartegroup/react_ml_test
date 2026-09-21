"""
Logging helpers.
"""
import logging
import os

_LOG_LEVEL_NAME = os.environ.get('MLT_LOG_LEVEL', default='INFO')
_LOG_LEVEL = getattr(logging, str(_LOG_LEVEL_NAME).strip().upper(), None)
_LOG_LEVEL = _LOG_LEVEL if isinstance(_LOG_LEVEL, int) else logging.INFO

_LOG_FORMAT = '%(levelname)s %(asctime)s: %(message)s'
_LOG_DATEFMT = '%H:%M:%S'

logger = logging.getLogger()

# NOTE: this must not be a bare logging.basicConfig(...) call, because basicConfig is a
# no-op whenever the root logger already has a handler - and the ML stack configures
# logging at import time, before this module is reached: the workflow scripts import
# mlptrain, which imports autode, whose log.py calls basicConfig() at CRITICAL (unless
# $AUTODE_LOG_LEVEL is set). Without force=True the level and format below were
# silently ignored, so wherever nothing else happened to lower the root level from
# CRITICAL every logger.info / logger.warning call in this package was dropped without
# a trace. (The optional 'coloredlogs' package, which mlptrain's log.py installs if it
# happens to be importable, is what otherwise pulls root down to INFO - so whether the
# logs appeared at all depended on an undeclared third-party package being installed.)
# force=True makes this configuration authoritative, whatever ran before it.
logging.basicConfig(
    level=_LOG_LEVEL,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATEFMT,
    force=True,
)
