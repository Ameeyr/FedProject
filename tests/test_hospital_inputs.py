from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_hospital_inputs_require_hospital_paths_or_uploaded_images():
    from app import validate_hospital_inputs

    assert validate_hospital_inputs([], []) is False
    assert validate_hospital_inputs(["/tmp/hospital_1"], []) is True
    assert validate_hospital_inputs([], ["image.png"]) is True


def test_real_hospital_dataset_has_both_classes_and_stratified_validation():
    from app import validate_hospital_dataset, split_hospital_train_validation

    dataset_root = Path(__file__).resolve().parents[2] / "dataset" / "hospital"

    for hospital_index in range(1, 4):
        hospital_dir = dataset_root / f"Hospital_{hospital_index}"
        summary = validate_hospital_dataset(hospital_dir)
        assert summary["healthy_count"] > 0
        assert summary["parkinson_count"] > 0
        assert summary["total_images"] == summary["healthy_count"] + summary["parkinson_count"]

        train_images, train_labels, val_images, val_labels = split_hospital_train_validation(
            summary["images"],
            summary["labels"],
            val_fraction=0.2,
            random_state=42,
        )

        assert len(train_images) > 0
        assert len(val_images) > 0
        assert np.sum(train_labels == 0) > 0
        assert np.sum(train_labels == 1) > 0
        assert np.sum(val_labels == 0) > 0
        assert np.sum(val_labels == 1) > 0


def test_federated_round_stores_three_hospital_histories_and_dashboard_metrics():
    from app import build_global_metrics
    from federated.server import FederatedFlowerServer

    class DummyModel:
        def __init__(self):
            self.weights = [np.array([1.0], dtype=np.float32)]

        def get_weights(self):
            return [np.array(weight, copy=True) for weight in self.weights]

        def set_weights(self, weights):
            self.weights = [np.array(weight, copy=True) for weight in weights]

        def evaluate(self, images, labels, verbose=0):
            return 0.75, 0.8

    class DummyClient:
        def __init__(self, client_id):
            self.client_id = client_id
            self.model = DummyModel()

        def train_model(self, train_data, val_data, epochs=1, batch_size=4, callbacks=None):
            train_images, train_labels = train_data
            val_images, val_labels = val_data
            assert len(train_images) > 0
            assert len(val_images) > 0
            return {
                "loss": [0.9, 0.7],
                "accuracy": [0.6, 0.8],
                "val_loss": [1.0, 0.9],
                "val_accuracy": [0.5, 0.7],
                "train_samples": int(len(train_images)),
                "val_samples": int(len(val_images)),
            }

    clients = [DummyClient(1), DummyClient(2), DummyClient(3)]
    client_datasets = {
        1: {"train": (np.ones((6, 1), dtype=np.float32), np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)),
            "val": (np.ones((2, 1), dtype=np.float32), np.array([0, 1], dtype=np.int32))},
        2: {"train": (np.ones((6, 1), dtype=np.float32), np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)),
            "val": (np.ones((2, 1), dtype=np.float32), np.array([0, 1], dtype=np.int32))},
        3: {"train": (np.ones((6, 1), dtype=np.float32), np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)),
            "val": (np.ones((2, 1), dtype=np.float32), np.array([0, 1], dtype=np.int32))},
    }
    server = FederatedFlowerServer(aggregation_mode="fedavg")
    server.run_federated_round(
        clients,
        round_train_fn=lambda client: (
            client.model.get_weights(),
            client.train_model(
                client_datasets[client.client_id]["train"],
                client_datasets[client.client_id]["val"],
                epochs=1,
                batch_size=4,
            ),
            len(client_datasets[client.client_id]["train"][0]),
        ),
        eval_data={1: client_datasets[1]["val"], 2: client_datasets[2]["val"], 3: client_datasets[3]["val"]},
    )

    assert len(server.client_histories) == 1
    assert len(server.client_histories[0]["local_histories"]) == 3
    assert len(server.global_round_metrics) == 1
    assert server.global_round_metrics[0]["round"] == 1

    dashboard_rows = build_global_metrics(server.global_round_metrics, clients_count=3)
    assert len(dashboard_rows) == 1
    assert dashboard_rows[0]["federated_round"] == 1
    assert dashboard_rows[0]["clients"] == 3
