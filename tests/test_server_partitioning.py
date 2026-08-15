import numpy as np
import pytest
import tensorflow as tf

from federated.server import (
    FederatedFlowerServer,
    build_flower_strategy,
    is_flower_available,
    partition_dataset,
)
from models.efficientnetb0 import FederatedClient


def test_partition_dataset_balances_class_labels():
    images = np.arange(12).reshape(12, 1, 1, 1)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)

    partitions = partition_dataset(images, labels, num_clients=3, seed=7)

    assert len(partitions) == 3
    for client_images, client_labels in partitions:
        assert client_images.shape[0] == client_labels.shape[0]
        assert client_labels.size > 0

    combined_labels = np.concatenate([client_labels for _, client_labels in partitions])
    assert np.array_equal(np.sort(combined_labels), np.sort(labels))


def test_local_training_returns_actual_keras_history_for_requested_epoch_count():
    class TinyClient(FederatedClient):
        def build_model(self, model_name, num_classes):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            return model

    client = TinyClient(client_id=1, server_address="local", model_name="tiny", num_classes=1)
    train_images = np.random.rand(16, 4).astype(np.float32)
    train_labels = np.array([0, 1] * 8, dtype=np.float32)
    val_images = np.random.rand(8, 4).astype(np.float32)
    val_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float32)

    history = client.train_model((train_images, train_labels), (val_images, val_labels), epochs=5, batch_size=4)

    assert isinstance(history, dict)
    assert len(history["loss"]) == 5
    assert len(history["accuracy"]) == 5
    assert len(history["val_loss"]) == 5
    assert len(history["val_accuracy"]) == 5


def test_federated_rounds_retrain_clients_between_rounds():
    class DummyModel:
        def __init__(self, base_value):
            self.base_value = float(base_value)
            self.weights = [np.array([self.base_value], dtype=np.float32)]

        def get_weights(self):
            return [np.array(weight, copy=True) for weight in self.weights]

        def set_weights(self, weights):
            self.weights = [np.array(weight, copy=True) for weight in weights]

        def evaluate(self, images, labels, verbose=0):
            return float(self.weights[0][0]), float(self.weights[0][0] + 0.1)

    class DummyClient:
        def __init__(self, base_value):
            self.client_id = base_value
            self.model = DummyModel(base_value)

    clients = [DummyClient(1), DummyClient(2)]
    server = FederatedFlowerServer(aggregation_mode="fedavg")

    def train_client_round(client):
        weights = client.model.get_weights()
        updated = [np.array([weight[0] + client.client_id], dtype=np.float32) for weight in weights]
        client.model.set_weights(updated)
        return updated, {"client_id": client.client_id, "round": server.round + 1}

    server.run_federated_training(clients, rounds=3, round_train_fn=train_client_round, eval_data=(np.ones((2, 1)), np.array([0, 1])))

    assert server.round == 3
    assert len(server.global_round_metrics) == 3
    assert server.global_round_metrics[0]["round"] == 1
    assert "loss" in server.global_round_metrics[0]
    assert "accuracy" in server.global_round_metrics[0]
    assert all(client.model.weights[0][0] > 0 for client in clients)


def test_global_evaluation_metrics_are_not_fabricated_as_validation_metrics():
    class DummyModel:
        def __init__(self):
            self.weights = [np.array([1.0], dtype=np.float32)]

        def get_weights(self):
            return [np.array(weight, copy=True) for weight in self.weights]

        def set_weights(self, weights):
            self.weights = [np.array(weight, copy=True) for weight in weights]

        def evaluate(self, images, labels, verbose=0):
            return 0.75, 0.80

    class DummyClient:
        def __init__(self):
            self.client_id = 1
            self.model = DummyModel()

    server = FederatedFlowerServer(aggregation_mode="fedavg")
    client = DummyClient()
    server.global_weights = client.model.get_weights()

    metrics = server.evaluate_global_model([client], (np.ones((2, 1)), np.array([0, 1])))

    assert metrics["loss"] == 0.75
    assert metrics["accuracy"] == 0.80
    assert "val_loss" not in metrics
    assert "val_accuracy" not in metrics


def test_global_evaluation_failure_is_reported_not_hidden():
    class FailingModel:
        def get_weights(self):
            return [np.array([1.0], dtype=np.float32)]

        def set_weights(self, weights):
            return None

        def evaluate(self, images, labels, verbose=0):
            raise RuntimeError("model evaluate failed")

    class DummyClient:
        def __init__(self):
            self.client_id = 1
            self.model = FailingModel()

    server = FederatedFlowerServer(aggregation_mode="fedavg")
    client = DummyClient()
    server.global_weights = client.model.get_weights()

    with pytest.raises(RuntimeError, match="Global evaluation failed.*Round 1"):
        server.evaluate_global_model([client], (np.ones((2, 1)), np.array([0, 1])))


def test_federated_round_requires_fresh_local_training_per_round():
    class DummyModel:
        def __init__(self, base_value):
            self.weights = [np.array([base_value], dtype=np.float32)]

        def get_weights(self):
            return [np.array(weight, copy=True) for weight in self.weights]

        def set_weights(self, weights):
            self.weights = [np.array(weight, copy=True) for weight in weights]

    class DummyClient:
        def __init__(self, client_id):
            self.client_id = client_id
            self.model = DummyModel(client_id)

    server = FederatedFlowerServer(aggregation_mode="fedavg")
    clients = [DummyClient(1), DummyClient(2)]
    server._initialize_global_weights(clients)

    with pytest.raises(ValueError, match="fresh local training|new local training"):
        server.run_federated_round(clients, round_train_fn=None)


def test_flower_support_is_optional_and_safe():
    assert isinstance(is_flower_available(), bool)

    if is_flower_available():
        strategy = build_flower_strategy("fedavg")
        assert strategy is not None
    else:
        try:
            build_flower_strategy("fedavg")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected a RuntimeError when Flower is unavailable")


def test_federated_round_error_reports_client_and_round():
    class DummyModel:
        def __init__(self, base_value):
            self.base_value = float(base_value)
            self.weights = [np.array([self.base_value], dtype=np.float32)]

        def get_weights(self):
            return [np.array(weight, copy=True) for weight in self.weights]

        def set_weights(self, weights):
            self.weights = [np.array(weight, copy=True) for weight in weights]

    class DummyClient:
        def __init__(self, client_id):
            self.client_id = client_id
            self.model = DummyModel(client_id)

    server = FederatedFlowerServer(aggregation_mode="fedavg")
    client = DummyClient(1)

    def train_client_round(_client):
        raise RuntimeError("local fit failed")

    with pytest.raises(RuntimeError, match="Client 1.*Round 1.*local fit"):
        server.run_federated_training([client], rounds=1, round_train_fn=train_client_round)
