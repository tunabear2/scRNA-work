"""
공통 로깅 유틸리티 — 콘솔 + 파일 동시 출력, 결과 JSON 저장
"""
import os
import sys
import json
import logging
import datetime


def get_logger(task_name: str, results_dir: str) -> logging.Logger:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "run.log")

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"=== {task_name} 시작: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"로그 파일: {log_path}")
    return logger


def save_metrics(metrics: dict, results_dir: str, filename: str = "metrics.json"):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    return path
