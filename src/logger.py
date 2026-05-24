"""logger.py — Centralizirana konfiguracija logginga za projekat."""

import logging
import sys


def _setup_root_logger() -> None:
    """Jedanput konfigurira root logger pri importu modula."""
    root = logging.getLogger()
    if root.handlers:
        return

    import io
    utf8_stdout = (
        io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        if hasattr(sys.stdout, "buffer")
        else sys.stdout
    )

    handler = logging.StreamHandler(utf8_stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Vraća logger s konzistentnim formatom za dati modul.

    Args:
        name: Ime loggera (obično __name__ modula koji poziva).

    Returns:
        Konfigurisani logging.Logger objekat.
    """
    _setup_root_logger()
    return logging.getLogger(name)
