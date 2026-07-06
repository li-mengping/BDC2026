"""统一日志工具；模块通过 `get_logger` 获取 stdout/文件双写日志。"""

import json
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """创建统一日志器，默认输出到 stdout，可选同步写文件。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_json(logger: logging.Logger, name: str, payload: dict) -> None:
    """统一记录结构化配置，便于实验复盘。"""
    logger.info('%s=%s', name, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
