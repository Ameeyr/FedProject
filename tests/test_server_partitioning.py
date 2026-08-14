import numpy as np
import pytest

from federated.server import (
    FederatedFlowerServer,
    build_flower_strategy,
    is_flower_available,
    partition_dataset,
)


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
