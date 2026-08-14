import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_METRICS_DB = Path(__file__).resolve().parent.parent / "result" / "training_metrics.db"


def _connect(db_path):
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def initialize_metrics_db(db_path=DEFAULT_METRICS_DB):
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT,
                aggregation_mode TEXT,
                config_json TEXT,
                history_by_client_json TEXT,
                aggregated_history_json TEXT,
                classification_metrics_json TEXT,
                accuracy REAL,
                val_accuracy REAL,
                loss REAL,
                val_loss REAL
            )
            """
        )
        columns = [row[1] for row in connection.execute("PRAGMA table_info(run_metrics)").fetchall()]
        if "classification_metrics_json" not in columns:
            connection.execute(
                "ALTER TABLE run_metrics ADD COLUMN classification_metrics_json TEXT"
            )
        connection.commit()
    finally:
        connection.close()


def save_run_metrics(
    db_path=DEFAULT_METRICS_DB,
    model_name=None,
    aggregation_mode=None,
    config=None,
    history_by_client=None,
    aggregated_history=None,
    classification_metrics=None,
    accuracy=None,
    val_accuracy=None,
    loss=None,
    val_loss=None,
):
    initialize_metrics_db(db_path)
    connection = _connect(db_path)
    try:
        row = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_name,
            aggregation_mode,
            json.dumps(_to_serializable(config or {}), sort_keys=True),
            json.dumps(_to_serializable(history_by_client or []), sort_keys=True),
            json.dumps(_to_serializable(aggregated_history or {}), sort_keys=True),
            json.dumps(_to_serializable(classification_metrics or {}), sort_keys=True),
            _to_serializable(accuracy),
            _to_serializable(val_accuracy),
            _to_serializable(loss),
            _to_serializable(val_loss),
        )
        cursor = connection.execute(
            """
            INSERT INTO run_metrics (
                created_at,
                model_name,
                aggregation_mode,
                config_json,
                history_by_client_json,
                aggregated_history_json,
                classification_metrics_json,
                accuracy,
                val_accuracy,
                loss,
                val_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        connection.commit()
        return cursor.lastrowid
    finally:
        connection.close()


def load_recent_metrics(db_path=DEFAULT_METRICS_DB, limit=10):
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT *
            FROM run_metrics
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
