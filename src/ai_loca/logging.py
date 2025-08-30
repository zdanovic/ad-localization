import logging
import os


def setup_logging() -> None:
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = os.environ.get(
        "LOG_FORMAT",
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    datefmt = os.environ.get("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

