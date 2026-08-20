import json
import sqlite3

from app import build_local_metrics, resolve_active_run_id
from metrics_store import initialize_metrics_db, load_recent_metrics, load_run_metrics_by_id, save_run_metrics


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


def test_build_local_metrics_preserves_actual_client_and_round_numbers():
    entries = [
        {"client_id": 1, "federated_round": 1, "history": {"loss": [1.0, 0.9], "accuracy": [0.5, 0.7], "val_loss": [1.1, 1.0], "val_accuracy": [0.4, 0.6]}},
        {"client_id": 1, "federated_round": 2, "history": {"loss": [0.8, 0.7], "accuracy": [0.8, 0.9], "val_loss": [0.9, 0.8], "val_accuracy": [0.7, 0.8]}},
        {"client_id": 2, "federated_round": 1, "history": {"loss": [1.2, 1.1], "accuracy": [0.4, 0.5], "val_loss": [1.3, 1.2], "val_accuracy": [0.3, 0.45]}},
    ]

    rows = build_local_metrics(entries, {1: "Hospital 1", 2: "Hospital 2"}, federated_round=99)

    assert sorted({row["federated_round"] for row in rows}) == [1, 2]
    assert {row["client"] for row in rows if row["federated_round"] == 1} == {"Hospital 1", "Hospital 2"}
    assert all(row["client"] == "Hospital 1" for row in rows if row["federated_round"] == 2)


def test_global_metrics_are_stored_one_row_per_federated_round(tmp_path):
    db_path = tmp_path / "global_round_metrics.db"
    run_id = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"rounds": 3},
        global_metrics=[
            {"round": 1, "global_loss": 0.90, "global_accuracy": 0.70, "global_val_loss": 1.10, "global_val_accuracy": 0.65, "clients": 2},
            {"round": 2, "global_loss": 0.80, "global_accuracy": 0.75, "global_val_loss": 1.00, "global_val_accuracy": 0.70, "clients": 2},
            {"round": 3, "global_loss": 0.70, "global_accuracy": 0.82, "global_val_loss": 0.90, "global_val_accuracy": 0.78, "clients": 2},
        ],
    )

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT federated_round, global_loss, global_accuracy FROM global_federated_metrics WHERE run_id = ? ORDER BY federated_round ASC",
            (run_id,),
        ).fetchall()
        assert len(rows) == 3
        assert [row[0] for row in rows] == [1, 2, 3]
    finally:
        connection.close()


def test_resolve_active_run_prefers_the_latest_run_id():
    assert resolve_active_run_id(["run-1", "run-2"], current_run_id="run-1", preferred_run_id="run-2") == "run-2"
    assert resolve_active_run_id(["run-1", "run-2"], current_run_id=None, preferred_run_id=None) == "run-1"


def test_global_metrics_without_validation_data_stay_unset(tmp_path):
    db_path = tmp_path / "global_metrics_without_validation.db"
    run_id = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"rounds": 2},
        global_metrics=[
            {"round": 1, "global_loss": 0.90, "global_accuracy": 0.70, "clients": 2},
            {"round": 2, "global_loss": 0.80, "global_accuracy": 0.75, "clients": 2},
        ],
    )

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT federated_round, global_val_loss, global_val_accuracy FROM global_federated_metrics WHERE run_id = ? ORDER BY federated_round ASC",
            (run_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] is None
        assert rows[0][2] is None
        assert rows[1][1] is None
        assert rows[1][2] is None
    finally:
        connection.close()


def test_save_run_metrics_round_trips_classification_metrics(tmp_path):
    db_path = tmp_path / "classification_metrics.db"
    classification_metrics = {
        "accuracy": 0.89,
        "precision": 0.88,
        "recall": 0.87,
        "f1_score": 0.86,
        "sensitivity": 0.85,
        "specificity": 0.84,
    }

    run_id = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"rounds": 1},
        classification_metrics=classification_metrics,
    )

    loaded = load_run_metrics_by_id(db_path=db_path, run_id=run_id)
    assert loaded is not None
    assert loaded["classification_metrics"]["accuracy"] == 0.89
    assert json.loads(loaded["classification_metrics_json"])["f1_score"] == 0.86


def test_save_run_metrics_stores_local_and_global_metrics_with_run_id(tmp_path):
    db_path = tmp_path / "run_metrics.db"

    first_run = save_run_metrics(
        db_path=db_path,
        model_name="efficientnetb0",
        aggregation_mode="fedavg",
        config={"rounds": 1},
        local_metrics=[
            {"client": "Hospital 1", "federated_round": 1, "epoch": 1, "loss": 1.2, "accuracy": 0.45, "val_loss": 1.3, "val_accuracy": 0.40},
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
            {"client": "Hospital 1", "federated_round": 2, "epoch": 2, "loss": 0.8, "accuracy": 0.75, "val_loss": 0.9, "val_accuracy": 0.70},
            {"client": "Hospital 2", "federated_round": 2, "epoch": 2, "loss": 0.7, "accuracy": 0.80, "val_loss": 0.8, "val_accuracy": 0.75},
        ],
        global_metrics=[
            {"round": 1, "global_loss": 1.0, "global_accuracy": 0.60, "global_val_loss": 1.1, "global_val_accuracy": 0.55, "clients": 2},
            {"round": 2, "global_loss": 0.6, "global_accuracy": 0.82, "global_val_loss": 0.7, "global_val_accuracy": 0.78, "clients": 2},
        ],
    )

    assert isinstance(first_run, str) and first_run.startswith("run-")
    assert isinstance(second_run, str) and second_run.startswith("run-")
    assert first_run != second_run

    latest = load_recent_metrics(db_path=db_path, limit=1)[0]
    assert latest["run_id"] == second_run
    assert latest["config_json"]

    connection = sqlite3.connect(db_path)
    try:
        local_columns = [row[1] for row in connection.execute("PRAGMA table_info(local_client_metrics)").fetchall()]
        global_columns = [row[1] for row in connection.execute("PRAGMA table_info(global_federated_metrics)").fetchall()]
        assert "federated_round" in local_columns
        assert "federated_round" in global_columns
        local_rows = connection.execute(
            "SELECT COUNT(*) FROM local_client_metrics WHERE run_id = ?",
            (second_run,),
        ).fetchone()[0]
        global_rows = connection.execute(
            "SELECT COUNT(*) FROM global_federated_metrics WHERE run_id = ?",
            (second_run,),
        ).fetchone()[0]
        assert local_rows == 2
        assert global_rows == 2

        first_run_local = connection.execute(
            "SELECT COUNT(*) FROM local_client_metrics WHERE run_id = ?",
            (first_run,),
        ).fetchone()[0]
        assert first_run_local == 1
    finally:
        connection.close()
