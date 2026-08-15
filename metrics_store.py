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


def _ensure_metric_rows(metrics):
    if metrics is None:
        return []
    if isinstance(metrics, dict):
        return [metrics]
    if isinstance(metrics, (list, tuple)):
        return [item for item in metrics if isinstance(item, dict)]
    return []


def initialize_metrics_db(db_path=DEFAULT_METRICS_DB):
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                model_name TEXT,
                aggregation_mode TEXT,
                config_json TEXT,
                history_by_client_json TEXT,
                aggregated_history_json TEXT,
                classification_metrics_json TEXT,
                local_metrics_json TEXT,
                global_metrics_json TEXT,
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
        if "local_metrics_json" not in columns:
            connection.execute(
                "ALTER TABLE run_metrics ADD COLUMN local_metrics_json TEXT"
            )
        if "global_metrics_json" not in columns:
            connection.execute(
                "ALTER TABLE run_metrics ADD COLUMN global_metrics_json TEXT"
            )
        if "run_id" not in columns:
            # keep old rows consistent by generating a synthetic identifier
            connection.execute("ALTER TABLE run_metrics ADD COLUMN run_id TEXT")
            connection.execute(
                "UPDATE run_metrics SET run_id = 'run-' || id WHERE run_id IS NULL"
            )
        connection.commit()

        connection.execute(
            "CREATE TABLE IF NOT EXISTS local_client_metrics (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                run_id TEXT NOT NULL,\
                client TEXT,\
                federated_round INTEGER,\
                epoch INTEGER,\
                loss REAL,\
                accuracy REAL,\
                val_loss REAL,\
                val_accuracy REAL,\
                created_at TEXT NOT NULL\
            )"
        )
        local_columns = [row[1] for row in connection.execute("PRAGMA table_info(local_client_metrics)").fetchall()]
        if "federated_round" not in local_columns:
            connection.execute("ALTER TABLE local_client_metrics ADD COLUMN federated_round INTEGER")

        connection.execute(
            "CREATE TABLE IF NOT EXISTS global_federated_metrics (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                run_id TEXT NOT NULL,\
                federated_round INTEGER,\
                global_loss REAL,\
                global_accuracy REAL,\
                global_val_loss REAL,\
                global_val_accuracy REAL,\
                clients INTEGER,\
                created_at TEXT NOT NULL\
            )"
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
    local_metrics=None,
    global_metrics=None,
    run_id=None,
):
    initialize_metrics_db(db_path)
    connection = _connect(db_path)
    try:
        if run_id is None:
            run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        row = (
            run_id,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_name,
            aggregation_mode,
            json.dumps(_to_serializable(config or {}), sort_keys=True),
            json.dumps(_to_serializable(history_by_client or []), sort_keys=True),
            json.dumps(_to_serializable(aggregated_history or {}), sort_keys=True),
            json.dumps(_to_serializable(classification_metrics or {}), sort_keys=True),
            json.dumps(_to_serializable(local_metrics or []), sort_keys=True),
            json.dumps(_to_serializable(global_metrics or []), sort_keys=True),
            _to_serializable(accuracy),
            _to_serializable(val_accuracy),
            _to_serializable(loss),
            _to_serializable(val_loss),
        )
        connection.execute(
            """
            INSERT INTO run_metrics (
                run_id,
                created_at,
                model_name,
                aggregation_mode,
                config_json,
                history_by_client_json,
                aggregated_history_json,
                classification_metrics_json,
                local_metrics_json,
                global_metrics_json,
                accuracy,
                val_accuracy,
                loss,
                val_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )

        normalized_local_metrics = _ensure_metric_rows(local_metrics)
        normalized_global_metrics = _ensure_metric_rows(global_metrics)

        connection.executemany(
            """
            INSERT INTO local_client_metrics (run_id, client, federated_round, epoch, loss, accuracy, val_loss, val_accuracy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.get("client"),
                    item.get("federated_round", item.get("round")),
                    item.get("epoch"),
                    item.get("loss"),
                    item.get("accuracy"),
                    item.get("val_loss"),
                    item.get("val_accuracy"),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                for item in normalized_local_metrics
            ],
        )

        connection.executemany(
            """
            INSERT INTO global_federated_metrics (run_id, federated_round, global_loss, global_accuracy, global_val_loss, global_val_accuracy, clients, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.get("federated_round", item.get("round")),
                    item.get("global_loss", item.get("loss")),
                    item.get("global_accuracy", item.get("accuracy")),
                    item.get("global_val_loss", item.get("val_loss")),
                    item.get("global_val_accuracy", item.get("val_accuracy")),
                    item.get("clients"),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                for item in normalized_global_metrics
            ],
        )

        connection.commit()
        return run_id
    finally:
        connection.close()


def load_run_metrics_by_id(db_path=DEFAULT_METRICS_DB, run_id=None):
    if not run_id:
        return None
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM run_metrics WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (str(run_id),),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        connection.close()


def load_run_local_metrics(db_path=DEFAULT_METRICS_DB, run_id=None):
    if not run_id:
        return []
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM local_client_metrics WHERE run_id = ? ORDER BY id ASC",
            (str(run_id),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def load_run_global_metrics(db_path=DEFAULT_METRICS_DB, run_id=None):
    if not run_id:
        return []
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM global_federated_metrics WHERE run_id = ? ORDER BY id ASC",
            (str(run_id),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def list_run_ids(db_path=DEFAULT_METRICS_DB, limit=20):
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT run_id FROM run_metrics WHERE run_id IS NOT NULL ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [str(row["run_id"]) for row in rows if row["run_id"]]
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


def load_latest_run_metrics(db_path=DEFAULT_METRICS_DB):
    rows = load_recent_metrics(db_path=db_path, limit=1)
    if not rows:
        return None
    return rows[0]
