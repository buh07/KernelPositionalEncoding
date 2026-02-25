from __future__ import annotations

import logging

_LOGGER_NAME = "kernel_pe"


def get_logger(name: str | None = None) -> logging.Logger:
    full_name = f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME
    logger = logging.getLogger(full_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
