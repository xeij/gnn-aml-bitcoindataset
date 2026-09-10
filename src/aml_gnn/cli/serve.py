"""Command-line entry point for the Gradio investigation app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..app.interface import APP_THEME, build_interface
from ..app.state import DEFAULT_CHECKPOINT_DIR
from ..logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``aml-app``."""
    parser = argparse.ArgumentParser(
        prog="aml-app",
        description="Launch the AML graph-investigation interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
        help="Directory scanned for trained checkpoints.",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint name to load on startup."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio tunnel. Off by default.",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="Also log to this file.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the Gradio app and return a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, log_file=args.log_file, force=True)

    demo = build_interface(args.checkpoint_dir, default_checkpoint=args.checkpoint)
    logger.info("Serving on http://%s:%d", args.host, args.port)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
        theme=APP_THEME,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
