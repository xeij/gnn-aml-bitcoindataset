"""Centralised logging configuration for the AML-GNN package.

A single :func:`configure_logging` call at process entry (CLI or Gradio app)
installs handlers on the ``aml_gnn`` root logger.  Library modules never
configure handlers themselves; they only ever call :func:`get_logger`.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Final, Optional

PACKAGE_LOGGER_NAME: Final[str] = "aml_gnn"

_CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_FILE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT: Final[str] = "%H:%M:%S"

_ANSI: Final[dict[int, str]] = {
    logging.DEBUG: "\033[38;5;244m",
    logging.INFO: "\033[38;5;39m",
    logging.WARNING: "\033[38;5;214m",
    logging.ERROR: "\033[38;5;196m",
    logging.CRITICAL: "\033[1;38;5;196m",
}
_RESET: Final[str] = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Console formatter that colourises the level name when attached to a TTY."""

    def __init__(self, fmt: str, datefmt: str, *, use_colour: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_colour:
            return super().format(record)
        # Copy so concurrent handlers do not observe the mutated level name.
        coloured = logging.makeLogRecord(record.__dict__)
        colour = _ANSI.get(record.levelno, "")
        coloured.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(coloured)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Optional[Path] = None,
    force: bool = False,
) -> logging.Logger:
    """Install console (and optionally rotating-file) handlers on the package logger.

    Args:
        level: Threshold for the package logger, as a level number or name.
        log_file: When given, logs are additionally written to this path with
            rotation at 10 MB across three backups.  Parent directories are created.
        force: Replace pre-existing handlers instead of returning early.  Useful
            when a notebook or test harness reconfigures logging.

    Returns:
        The configured ``aml_gnn`` logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if logger.handlers and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(level)
    # The package logger is a terminal sink: do not leak records to the root
    # logger, whose handlers are owned by whatever host process imports us.
    logger.propagate = False

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        _ColourFormatter(
            _CONSOLE_FORMAT,
            _DATE_FORMAT,
            use_colour=sys.stdout.isatty(),
        )
    )
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)
        logger.debug("File logging enabled at %s", log_file)

    # Third-party loggers are noisy at INFO during dataset download / UI startup.
    for noisy in ("httpx", "urllib3", "matplotlib", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package logger.

    Args:
        name: Usually ``__name__``.  A fully-qualified module name inside the
            package is used as-is; anything else is nested under ``aml_gnn``.
    """
    if name == PACKAGE_LOGGER_NAME or name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")
