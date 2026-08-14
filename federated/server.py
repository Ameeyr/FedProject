import copy
from pathlib import Path

import numpy as np


def partition_dataset(images, labels, num_clients=2, seed=42):
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if len(images) < num_clients:
        raise ValueError("Not enough samples for the requested number of clients")

    rng = np.random.default_rng(seed)
    class_ids = np.unique(labels)
    partitions = []

    for class_id in class_ids:
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        class_chunks = np.array_split(class_indices, num_clients)
        for client_idx, chunk in enumerate(class_chunks):
            if client_idx >= len(partitions):
                partitions.append([])
            partitions[client_idx].extend(chunk.tolist())

    client_splits = []
    for client_idx in range(num_clients):
        indices = np.array(partitions[client_idx], dtype=np.int64)
        if len(indices) == 0:
            fallback = np.flatnonzero(labels == labels[0])[0]
            indices = np.array([fallback], dtype=np.int64)
        client_splits.append(indices)

    result = []
    for indices in client_splits:
        result.append((images[indices], labels[indices]))
    return result


class FederatedFlowerServer:
    def __init__(self, model_name="efficientnetb0", aggregation_mode="fedavg", mu=0.01):
        self.model_name = model_name
        self.aggregation_mode = aggregation_mode.lower()
        self.mu = float(mu)
        self.round = 0
        self.global_weights = None
        self.history = []
        self.client_histories = []

    def _validate_aggregation_mode(self):
        if self.aggregation_mode not in {"fedavg", "fedprox"}:
            raise ValueError("aggregation_mode must be either 'fedavg' or 'fedprox'")

    def update_global_model(self, client_weights):
        self._validate_aggregation_mode()
        if client_weights is None:
            raise ValueError("Client weights are required for a federated update")

        if self.global_weights is None:
            self.global_weights = [np.array(weight, copy=True) for weight in client_weights]
        else:
            if self.aggregation_mode == "fedavg":
                self.global_weights = [
                    (self.global_weights[idx] * self.round + np.array(weight, copy=True)) / (self.round + 1)
                    for idx, weight in enumerate(client_weights)
                ]
            else:
                self.global_weights = [
                    (self.global_weights[idx] * self.round + np.array(weight, copy=True)) / (self.round + 1)
                    for idx, weight in enumerate(client_weights)
                ]

        self.round += 1
        self.history.append({"round": self.round, "weights_shape": [np.shape(weight) for weight in self.global_weights], "mode": self.aggregation_mode})
        return self.global_weights

    def aggregate_client_updates(self, client_weights_list):
        if not client_weights_list:
            raise ValueError("At least one client weight set is required")

        averaged = []
        for idx in range(len(client_weights_list[0])):
            stacked = np.stack([np.array(weights[idx], copy=True) for weights in client_weights_list], axis=0)
            averaged.append(np.mean(stacked, axis=0))
        return averaged

    def aggregate_with_fedprox(self, client_weights_list, global_weights):
        self._validate_aggregation_mode()
        if self.aggregation_mode != "fedprox":
            return self.aggregate_client_updates(client_weights_list)

        if global_weights is None:
            return self.aggregate_client_updates(client_weights_list)

        proximal_updates = []
        for idx in range(len(client_weights_list[0])):
            stacked = np.stack([np.array(weights[idx], copy=True) for weights in client_weights_list], axis=0)
            averaged = np.mean(stacked, axis=0)
            global_arr = np.array(global_weights[idx], copy=True)
            proximal_updates.append(averaged + self.mu * (averaged - global_arr))
        return proximal_updates

    def run_federated_round(self, clients):
        client_weights = []
        for client in clients:
            client_weights.append(client.model.get_weights())

        aggregated_weights = self.aggregate_with_fedprox(client_weights, self.global_weights)
        self.update_global_model(aggregated_weights)
        self.client_histories.append({"round": self.round, "clients": len(clients), "mode": self.aggregation_mode})

        for client in clients:
            client.model.set_weights(self.global_weights)
        return self.global_weights

    def apply_global_update(self, client):
        if self.global_weights is None:
            return None
        client.model.set_weights(self.global_weights)
        return self.global_weights

    def save_state(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, *self.global_weights)

    def load_state(self, output_path):
        output_path = Path(output_path)
        if not output_path.exists():
            return None
        loaded = np.load(output_path, allow_pickle=True)
        self.global_weights = [loaded[name] for name in loaded.files]
        self.round = max(1, len(self.history) + 1)
        return self.global_weights
