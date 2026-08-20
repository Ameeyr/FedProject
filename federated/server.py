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

            if isinstance(dataset, dict):
                train_data = dataset.get("train") or dataset.get("train_data")
                val_data = dataset.get("val") or dataset.get("validation") or dataset.get("val_data")
                if train_data is None or val_data is None:
                    raise ValueError(f"Client {client.client_id} dataset is missing train or validation data.")
            elif len(dataset) == 2:
                train_data = (dataset[0], dataset[1])
                val_data = None
                if client_datasets is not None and isinstance(client_datasets, dict):
                    val_data = client_datasets.get(client.client_id, {}).get("val")
                if val_data is None:
                    _, local_val = server._split_dataset_for_local_validation(dataset[0], dataset[1], client)
                    val_data = local_val
            else:
                train_data = (dataset[0], dataset[1])
                val_data = (dataset[2], dataset[3])

            client.model.set_weights(parameters_to_ndarrays(current_parameters))
            client.train_model(
                train_data,
                val_data,
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
                        int(len(train_data[0])),
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

    def aggregate_client_updates(self, client_weights_list, sample_counts=None):
        if not client_weights_list:
            raise ValueError("At least one client weight set is required")

        if sample_counts is None:
            sample_counts = [1] * len(client_weights_list)
        sample_counts = [max(1, int(count)) for count in sample_counts]
        total_samples = float(sum(sample_counts))
        weights = [count / total_samples for count in sample_counts]

        averaged = []
        for idx in range(len(client_weights_list[0])):
            stacked = np.stack([np.array(weights_list[idx], copy=True) for weights_list in client_weights_list], axis=0)
            averaged.append(np.tensordot(np.asarray(weights, dtype=np.float64), stacked, axes=([0], [0])))
        return averaged

    def aggregate_with_fedprox(self, client_weights_list, global_weights, sample_counts=None):
        self._validate_aggregation_mode()
        if self.aggregation_mode != "fedprox":
            return self.aggregate_client_updates(client_weights_list, sample_counts=sample_counts)

        if global_weights is None:
            return self.aggregate_client_updates(client_weights_list, sample_counts=sample_counts)

        weighted_averaged = self.aggregate_client_updates(client_weights_list, sample_counts=sample_counts)
        proximal_updates = []
        for idx in range(len(client_weights_list[0])):
            global_arr = np.array(global_weights[idx], copy=True)
            proximal_updates.append(weighted_averaged[idx] + self.mu * (weighted_averaged[idx] - global_arr))
        return proximal_updates

    def evaluate_global_model(self, clients, eval_data):
        if not clients or eval_data is None:
            return None

        if isinstance(eval_data, dict):
            local_metrics = []
            total_validation_samples = 0
            hospital_diagnostics = []
            for client in clients:
                client_id = getattr(client, "client_id", None)
                if client_id not in eval_data:
                    raise ValueError(
                        f"Missing local validation data for client '{client_id}' during global evaluation. "
                        "Only numerical metrics should be returned to the server."
                    )

                client_eval_data = eval_data[client_id]
                if client_eval_data is None:
                    raise ValueError(f"Local validation data for client '{client_id}' is empty.")

                eval_images, eval_labels = client_eval_data
                if eval_images is None or eval_labels is None:
                    raise ValueError(f"Local validation data for client '{client_id}' is incomplete.")

                sample_count = int(len(eval_images) if hasattr(eval_images, "__len__") else len(eval_labels))
                if sample_count <= 0:
                    raise ValueError(f"Local validation data for client '{client_id}' contains no samples.")

                if self.global_weights is not None:
                    client.model.set_weights(self.global_weights)

                try:
                    evaluation_result = client.model.evaluate(eval_images, eval_labels, verbose=0)
                    if isinstance(evaluation_result, dict):
                        loss_value = float(evaluation_result.get("loss", evaluation_result.get("val_loss", 0.0)))
                        accuracy_value = float(evaluation_result.get("accuracy", evaluation_result.get("val_accuracy", 0.0)))
                    elif isinstance(evaluation_result, (list, tuple)) and len(evaluation_result) >= 2:
                        loss_value = float(evaluation_result[0])
                        accuracy_value = float(evaluation_result[1])
                    else:
                        raise TypeError("Model evaluation did not return loss and accuracy values.")
                except Exception as exc:
                    raise RuntimeError(
                        f"Global evaluation failed during Round {self.round + 1} for client {client_id}: {exc}"
                    ) from exc

                total_validation_samples += sample_count
                local_metrics.append({
                    "loss": float(loss_value),
                    "accuracy": float(accuracy_value),
                    "samples": sample_count,
                })
                hospital_diagnostics.append({
                    "round": self.round,
                    "hospital": f"Hospital {client_id}",
                    "loss": float(loss_value),
                    "accuracy": float(accuracy_value),
                    "validation_samples": sample_count,
                })

            if not local_metrics:
                return None

            weighted_loss = sum(item["loss"] * item["samples"] for item in local_metrics) / total_validation_samples
            weighted_accuracy = sum(item["accuracy"] * item["samples"] for item in local_metrics) / total_validation_samples
            metrics = {
                "round": self.round,
                "loss": float(weighted_loss),
                "accuracy": float(weighted_accuracy),
                "clients": len(local_metrics),
                "local_validation_samples": total_validation_samples,
                "hospital_metrics": hospital_diagnostics,
            }
            self.global_round_metrics.append(metrics)
            return metrics

        eval_images, eval_labels = eval_data
        if eval_images is None or eval_labels is None:
            return None

        model_client = clients[0]
        if self.global_weights is not None:
            model_client.model.set_weights(self.global_weights)

        try:
            loss, accuracy = model_client.model.evaluate(eval_images, eval_labels, verbose=0)
        except Exception as exc:
            raise RuntimeError(
                f"Global evaluation failed during Round {self.round + 1}: {exc}"
            ) from exc

        metrics = {
            "round": self.round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "clients": len(clients),
        }

        self.global_round_metrics.append(metrics)
        return metrics

    def run_federated_round(self, clients, round_train_fn=None, eval_data=None):
        if not clients:
            return None

        if round_train_fn is None:
            raise ValueError(
                "Federated rounds require fresh local training after each client receives the latest global model. "
                "Provide a round_train_fn that retrains each client before aggregation."
            )

        try:
            self._initialize_global_weights(clients)
            local_weights = []
            local_histories = []

            for client in clients:
                try:
                    client.model.set_weights(self.global_weights)
                    round_result = round_train_fn(client)
                    if len(round_result) == 3:
                        trained_weights, local_history, sample_count = round_result
                    elif len(round_result) == 2:
                        trained_weights, local_history = round_result
                        sample_count = 1
                    else:
                        raise ValueError("round_train_fn must return (weights, history) or (weights, history, sample_count)")
                    local_weights.append(trained_weights)
                    local_histories.append({
                        "client_id": getattr(client, "client_id", "unknown"),
                        "federated_round": self.round + 1,
                        "history": local_history,
                        "sample_count": int(sample_count),
                    })
                except Exception as exc:
                    raise RuntimeError(
                        f"Federated training failed for Client {getattr(client, 'client_id', 'unknown')} "
                        f"during Round {self.round + 1}: local training/update operation failed: {exc}"
                    ) from exc

            sample_counts = [int(item.get("sample_count", 1)) for item in local_histories]
            aggregated_weights = self.aggregate_with_fedprox(local_weights, self.global_weights, sample_counts=sample_counts)
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

    @staticmethod
    def _split_dataset_for_local_validation(images, labels, client=None):
        if len(images) < 2:
            return (images, labels), (images, labels)

        seed = 42 + int(getattr(client, "client_id", 0))
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(images))
        split_index = max(1, int((1 - 0.2) * len(images)))
        train_idx = indices[:split_index]
        val_idx = indices[split_index:]
        return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])

    def apply_global_update(self, client):
        if self.global_weights is None:
            return None
        client.model.set_weights(self.global_weights)
        return self.global_weights

    def save_state(self, output_path):
        output_path = Path(output_path)
        if self.global_weights is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "round": np.asarray(int(self.round), dtype=np.int64),
            "aggregation_mode": np.asarray(str(self.aggregation_mode), dtype="U32"),
            "history": np.asarray(self.history, dtype=object),
        }
        for idx, weight in enumerate(self.global_weights):
            payload[f"weight_{idx}"] = np.asarray(weight)

        np.savez(output_path, **payload)
        return output_path

    def load_state(self, output_path):
        output_path = Path(output_path)
        if not output_path.exists():
            return None

        with np.load(output_path, allow_pickle=True) as loaded:
            weight_keys = [key for key in loaded.files if key.startswith("weight_")]
            if weight_keys:
                self.global_weights = [np.asarray(loaded[key]) for key in sorted(weight_keys, key=lambda key: int(key.split("_")[-1]))]
            else:
                self.global_weights = None

            if "round" in loaded.files:
                self.round = int(np.asarray(loaded["round"]).item())
            else:
                self.round = max(1, len(self.history) + 1)

            if "history" in loaded.files:
                history_value = loaded["history"]
                if isinstance(history_value, np.ndarray) and history_value.dtype == object:
                    self.history = history_value.tolist()
                else:
                    self.history = history_value.tolist() if hasattr(history_value, "tolist") else list(history_value)
            elif self.history:
                self.history = self.history
            else:
                self.history = []

            if "aggregation_mode" in loaded.files:
                mode_value = np.asarray(loaded["aggregation_mode"]).item()
                self.aggregation_mode = str(mode_value).lower() if mode_value is not None else self.aggregation_mode

        return self.global_weights
