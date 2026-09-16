import logging

from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "mojo_opset")
