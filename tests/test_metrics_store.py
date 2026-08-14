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
