import copy
from pathlib import Path

import numpy as np

try:
    import flwr as fl
    from flwr.common import Code, FitRes, Parameters, Status, ndarrays_to_parameters, parameters_to_ndarrays
except Exception:  # pragma: no cover - optional dependency
    fl = None
    Code = None
    FitRes = None
    Parameters = None
    Status = None
    ndarrays_to_parameters = None
    parameters_to_ndarrays = None


def is_flower_available():
    return fl is not None


def build_flower_strategy(aggregation_mode="fedavg", mu=0.01):
    if not is_flower_available():
        raise RuntimeError(
            "Flower is not installed in the current environment. "
            "Install 'flwr' to enable the Flower backend. "
            "The custom federated pipeline remains available as the default fallback."
        )

    mode = str(aggregation_mode).lower()
    if mode == "fedavg":
        from flwr.server.strategy import FedAvg
        return FedAvg(min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1)

    if mode == "fedprox":
        from flwr.server.strategy import FedProx
        return FedProx(
            proximal_mu=float(mu),
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

    raise ValueError("aggregation_mode must be either 'fedavg' or 'fedprox'")


class DummyFlowerProxy:
    def __init__(self, client_id):
        self.cid = str(client_id)


def run_flower_federation(
    clients,
    server,
    rounds=1,
    client_datasets=None,
    epochs=5,
    batch_size=32,
    callbacks=None,
    eval_data=None,
    aggregation_mode="fedavg",
    mu=0.01,
):
    if not is_flower_available():
        raise RuntimeError("Flower backend selected, but the flwr package is not installed.")

    if not clients:
        return server

    strategy = build_flower_strategy(aggregation_mode=aggregation_mode, mu=mu)
    current_parameters = ndarrays_to_parameters(clients[0].model.get_weights())
    strategy.initial_parameters = current_parameters

    for round_idx in range(1, int(rounds) + 1):
        fit_results = []
        for client in clients:
            dataset = None
            if client_datasets is not None:
                dataset = client_datasets.get(client.client_id, client_datasets.get(client.client_id - 1, None))
                if dataset is None:
                    dataset = client_datasets.get("default")
            if dataset is None:
                continue

            client_images, client_labels = dataset
            client.model.set_weights(parameters_to_ndarrays(current_parameters))
            client.train_model(
                (client_images, client_labels),
                (client_images, client_labels),
                epochs=int(epochs),
                batch_size=int(batch_size),
                callbacks=callbacks or [],
            )
            fit_results.append(
                (
                    DummyFlowerProxy(client.client_id),
                    FitRes(
                        Status(Code.OK, "ok"),
                        ndarrays_to_parameters(client.model.get_weights()),
                        int(len(client_images)),
                        {},
                    ),
                )
            )

        if not fit_results:
            continue

        aggregated_params, metrics = strategy.aggregate_fit(round_idx, fit_results, failures=[])
        if aggregated_params is not None:
            current_parameters = aggregated_params
            server.global_weights = parameters_to_ndarrays(current_parameters)
            server.round = round_idx
            server.history.append({
                "round": server.round,
                "weights_shape": [np.shape(weight) for weight in server.global_weights],
                "mode": aggregation_mode,
            })

        for client in clients:
            client.model.set_weights(server.global_weights)

        if eval_data is not None:
            server.evaluate_global_model(clients, eval_data)

    return server


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
        self.global_round_metrics = []

    def _validate_aggregation_mode(self):
        if self.aggregation_mode not in {"fedavg", "fedprox"}:
            raise ValueError("aggregation_mode must be either 'fedavg' or 'fedprox'")

    def _initialize_global_weights(self, clients):
        if not clients:
            return None
        if self.global_weights is None:
            seed_weights = [np.array(weight, copy=True) for weight in clients[0].model.get_weights()]
            self.global_weights = seed_weights
        for client in clients:
            client.model.set_weights(self.global_weights)
        return self.global_weights

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

    def evaluate_global_model(self, clients, eval_data):
        if not clients or eval_data is None:
            return None

        eval_images, eval_labels = eval_data
        if eval_images is None or eval_labels is None:
            return None

        model_client = clients[0]
        if self.global_weights is not None:
            model_client.model.set_weights(self.global_weights)

        loss, accuracy = model_client.model.evaluate(eval_images, eval_labels, verbose=0)
        metrics = {
            "round": self.round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "val_loss": float(loss),
            "val_accuracy": float(accuracy),
            "clients": len(clients),
        }
        self.global_round_metrics.append(metrics)
        return metrics

    def run_federated_round(self, clients, round_train_fn=None, eval_data=None):
        if not clients:
            return None

        try:
            self._initialize_global_weights(clients)
            local_weights = []
            local_histories = []

            if round_train_fn is None:
                for client in clients:
                    client.model.set_weights(self.global_weights)
                    local_weights.append(client.model.get_weights())
            else:
                for client in clients:
                    try:
                        client.model.set_weights(self.global_weights)
                        trained_weights, local_history = round_train_fn(client)
                        local_weights.append(trained_weights)
                        local_histories.append(local_history)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Federated training failed for Client {getattr(client, 'client_id', 'unknown')} "
                            f"during Round {self.round + 1}: local training/update operation failed: {exc}"
                        ) from exc

            aggregated_weights = self.aggregate_with_fedprox(local_weights, self.global_weights)
            self.update_global_model(aggregated_weights)
            self.client_histories.append({
                "round": self.round,
                "clients": len(clients),
                "mode": self.aggregation_mode,
                "local_histories": local_histories,
            })

            for client in clients:
                client.model.set_weights(self.global_weights)

            if eval_data is not None:
                self.evaluate_global_model(clients, eval_data)
            return self.global_weights
        except Exception as exc:
            raise RuntimeError(
                f"Federated round failed during round {self.round + 1}: {exc}"
            ) from exc

    def run_federated_training(self, clients, rounds=1, round_train_fn=None, eval_data=None):
        if rounds <= 0:
            return self.global_weights

        for round_index in range(1, int(rounds) + 1):
            try:
                self.round = round_index - 1
                self.run_federated_round(clients, round_train_fn=round_train_fn, eval_data=eval_data)
            except Exception as exc:
                if "Client" in str(exc) or "Round" in str(exc):
                    raise RuntimeError(str(exc)) from exc
                raise RuntimeError(
                    f"Federated training failed during Round {round_index}: {exc}"
                ) from exc
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
