"""结构化日志模块。基于标准库 logging，零外部依赖。

用法:
    from log_utils import get_logger
    logger = get_logger(__name__)
    logger.info("请求处理完成: %s", detail)
"""

import logging
import os
import sys

_default_level = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging(level: str | None = None) -> None:
    """全局日志初始化（在应用入口调用一次）。"""
    lvl = getattr(logging, (level or _default_level), logging.INFO)
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    logging.basicConfig(
        level=lvl,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(name)
