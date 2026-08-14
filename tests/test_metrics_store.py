import json
import sqlite3

from metrics_store import initialize_metrics_db, load_recent_metrics, save_run_metrics


def test_initialize_metrics_db_migrates_existing_table_without_new_column(tmp_path):
    db_path = tmp_path / "legacy_metrics.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE run_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT,
                aggregation_mode TEXT,
                config_json TEXT,
                history_by_client_json TEXT,
                aggregated_history_json TEXT,
                accuracy REAL,
                val_accuracy REAL,
                loss REAL,
                val_loss REAL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    initialize_metrics_db(db_path)

    connection = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(run_metrics)").fetchall()]
        assert "classification_metrics_json" in columns
    finally:
        connection.close()


def test_save_and_load_training_metrics(tmp_path):
    history_by_client = [
        {"loss": [1.0, 0.5], "accuracy": [0.2, 0.8]},
        {"loss": [0.9, 0.4], "accuracy": [0.3, 0.9]},
    ]
    aggregated_history = {"loss": [0.95, 0.45], "accuracy": [0.25, 0.85]}

    db_path = tmp_path / "training_metrics.db"
    run_id = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"epochs": 2, "batch_size": 4},
        history_by_client=history_by_client,
        aggregated_history=aggregated_history,
    )

    rows = load_recent_metrics(db_path=db_path, limit=5)

    assert run_id is not None
    assert len(rows) == 1
    assert rows[0]["model_name"] == "efficientnetb0"
    assert rows[0]["aggregation_mode"] == "fedavg"
    assert json.loads(rows[0]["history_by_client_json"])[0]["loss"] == [1.0, 0.5]
    assert json.loads(rows[0]["aggregated_history_json"])["accuracy"] == [0.25, 0.85]


def test_save_run_metrics_stores_local_and_global_metrics_with_run_id(tmp_path):
    db_path = tmp_path / "run_metrics.db"

    first_run = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"rounds": 1},
        local_metrics=[
            {"client": "Hospital 1", "epoch": 1, "loss": 1.2, "accuracy": 0.45, "val_loss": 1.3, "val_accuracy": 0.40},
        ],
        global_metrics=[
            {"round": 1, "global_loss": 1.2, "global_accuracy": 0.45, "global_val_loss": 1.3, "global_val_accuracy": 0.40, "clients": 1},
        ],
    )

    second_run = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"rounds": 2},
        local_metrics=[
            {"client": "Hospital 1", "epoch": 2, "loss": 0.8, "accuracy": 0.75, "val_loss": 0.9, "val_accuracy": 0.70},
            {"client": "Hospital 2", "epoch": 2, "loss": 0.7, "accuracy": 0.80, "val_loss": 0.8, "val_accuracy": 0.75},
        ],
        global_metrics=[
            {"round": 1, "global_loss": 1.0, "global_accuracy": 0.60, "global_val_loss": 1.1, "global_val_accuracy": 0.55, "clients": 2},
            {"round": 2, "global_loss": 0.6, "global_accuracy": 0.82, "global_val_loss": 0.7, "global_val_accuracy": 0.78, "clients": 2},
        ],
    )

    latest = load_recent_metrics(db_path=db_path, limit=1)[0]
    assert second_run == latest["id"]
    assert latest["config_json"]
    assert latest["id"] == second_run
