# -*- coding: utf-8 -*-
"""统一日志入口。

日志同时输出到控制台和 logs/policy_rag.log，按天切分，保留 14 天。
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from core.config import config

_LOGGER_NAME = "policy_rag"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s.%(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = TimedRotatingFileHandler(
        config.LOG_FILE, when="midnight", backupCount=14, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = _build_logger()
