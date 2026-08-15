import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from dashboard_utils import (
    aggregate_training_history,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_training_history_by_client,
)
from metrics_store import (
    DEFAULT_METRICS_DB,
    list_run_ids,
    load_recent_metrics,
    load_run_global_metrics,
    load_run_local_metrics,
    load_run_metrics_by_id,
    save_run_metrics,
)
from models.efficientnetb0 import FederatedClient, SUPPORTED_MODELS
from preprocessing.preprocess_images import ImagePreprocessor
from federated.server import (
    FederatedFlowerServer,
    is_flower_available,
    partition_dataset,
    run_flower_federation,
)
from explainability.gradcam import GradCAM


st.set_page_config(page_title="Federated Medical Imaging Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_PATH = WORKSPACE_ROOT / "dataset"
DEFAULT_RESULT_DIR = WORKSPACE_ROOT / "result"


def resolve_dataset_path(dataset_path, uploaded_files):
    raw_path = (dataset_path or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if uploaded_files:
        return None

    raise FileNotFoundError(f"Dataset path not found: {raw_path}")


def _reset_run_logs():
    st.session_state.run_logs = []


def _append_run_log(message):
    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []
    st.session_state.run_logs.append(message)
    if len(st.session_state.run_logs) > 200:
        st.session_state.run_logs = st.session_state.run_logs[-200:]


def _render_run_log(log_placeholder):
    logs = st.session_state.get("run_logs", [])
    log_text = "\n".join(logs) if logs else "No log messages yet."
    log_placeholder.code(log_text, language="text")


def st_image_compat(image, **kwargs):
    try:
        return st.image(image, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'use_container_width'" not in str(exc):
            raise
        kwargs.pop("use_container_width", None)
        return st.image(image, **kwargs)


def st_dataframe_compat(data, **kwargs):
    try:
        return st.dataframe(data, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'use_container_width'" not in str(exc):
            raise
        kwargs.pop("use_container_width", None)
        return st.dataframe(data, **kwargs)


class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, status_placeholder, progress_bar, log_placeholder):
        super().__init__()
        self.status_placeholder = status_placeholder
        self.progress_bar = progress_bar
        self.log_placeholder = log_placeholder
        self.total_epochs = 1

    def _log(self, message):
        _append_run_log(message)
        _render_run_log(self.log_placeholder)

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", 1)
        self.status_placeholder.info("Training started. Progress updates will appear below.")
        self._log("Training started.")

    def on_epoch_begin(self, epoch, logs=None):
        self.status_placeholder.info(f"Starting epoch {epoch + 1}/{self.total_epochs}...")
        self._log(f"Starting epoch {epoch + 1}/{self.total_epochs}...")
        if self.total_epochs:
            self.progress_bar.progress(epoch / self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        accuracy = logs.get("accuracy", logs.get("acc", float("nan")))
        val_accuracy = logs.get("val_accuracy", logs.get("val_acc", float("nan")))
        message = (
            f"Epoch {epoch + 1}/{self.total_epochs}: loss={loss:.4f}, val_loss={val_loss:.4f}, "
            f"accuracy={accuracy:.4f}, val_accuracy={val_accuracy:.4f}"
        )
        self.status_placeholder.write(message)
        self._log(message)
        if self.total_epochs:
            self.progress_bar.progress((epoch + 1) / self.total_epochs)


@st.cache_resource(show_spinner=False)
def build_client(model_name, num_classes):
    return FederatedClient(client_id=1, server_address="local", model_name=model_name, num_classes=num_classes)


def parse_hospital_dataset_paths(raw_value):
    if raw_value is None:
        return []
    paths = [line.strip() for line in str(raw_value).splitlines() if line.strip()]
    return [path for path in paths if path]


def remap_labels_to_shared_classes(labels, class_names, shared_class_names):
    if not class_names or not shared_class_names:
        return labels
    mapping = {class_name: idx for idx, class_name in enumerate(shared_class_names)}
    remapped = np.array(labels, copy=True)
    for original_class, original_index in enumerate(class_names):
        if original_class in mapping:
            remapped[remapped == original_index] = mapping[original_class]
    return remapped.astype(np.int32)


def _normalize_class_labels(uploaded_files, class_label_input):
    if class_label_input:
        labels = [label.strip() for label in class_label_input.split(",") if label.strip()]
    else:
        labels = []

    if not labels:
        return [Path(uploaded.name).stem for uploaded in uploaded_files]

    if len(labels) == 1 and len(uploaded_files) > 1:
        labels = labels * len(uploaded_files)
    else:
        repeat_count = max(1, int(np.ceil(len(uploaded_files) / len(labels))))
        labels = (labels * repeat_count)[: len(uploaded_files)]

    return labels


def _prepare_uploaded_dataset(uploaded_files, class_label_input, base_dir):
    class_labels = _normalize_class_labels(uploaded_files, class_label_input)
    for uploaded, class_name in zip(uploaded_files, class_labels):
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / uploaded.name
        destination.write_bytes(uploaded.getvalue())
    return class_labels


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_files, target_size=(224, 224), batch_size=8, chunk_size=256, class_label_input=None, dataset_path=None, result_dir=None):
    resolved_dataset_path = resolve_dataset_path(dataset_path, uploaded_files)
    resolved_result_dir = str(Path(result_dir or DEFAULT_RESULT_DIR).expanduser())

    if resolved_dataset_path is not None:
        preprocessor = ImagePreprocessor(
            resolved_dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            output_dir=resolved_result_dir,
            chunk_size=chunk_size,
        )
        images, labels, class_names = preprocessor.preprocess_images()
        return images, labels, class_names

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _prepare_uploaded_dataset(uploaded_files, class_label_input, tmp_path)

        preprocessor = ImagePreprocessor(
            str(tmp_path),
            target_size=target_size,
            batch_size=batch_size,
            output_dir=resolved_result_dir,
            chunk_size=chunk_size,
        )
        images, labels, class_names = preprocessor.preprocess_images()
        return images, labels, class_names


def load_hospital_datasets(hospital_paths, target_size=(224, 224), batch_size=8, chunk_size=256, result_dir=None):
    datasets = []
    shared_class_names = None
    for hospital_path in hospital_paths:
        images, labels, class_names = load_dataset(
            uploaded_files=[],
            target_size=target_size,
            batch_size=batch_size,
            chunk_size=chunk_size,
            dataset_path=hospital_path,
            result_dir=result_dir,
        )
        if shared_class_names is None:
            shared_class_names = class_names
        else:
            labels = remap_labels_to_shared_classes(labels, class_names, shared_class_names)
        datasets.append({"path": hospital_path, "images": images, "labels": labels, "class_names": shared_class_names})
    return datasets


def summarize_hospital_datasets(hospital_paths, result_dir=None):
    summary = []
    for hospital_path in hospital_paths:
        hospital_dir = Path(hospital_path).expanduser()
        if not hospital_dir.exists():
            summary.append({"path": hospital_path, "exists": False, "images": 0, "classes": []})
            continue

        class_names = []
        image_count = 0
        for child in sorted(hospital_dir.iterdir()):
            if child.is_dir():
                class_names.append(child.name)
                image_count += sum(1 for _ in child.rglob("*") if _.is_file() and _.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"})
        summary.append({"path": str(hospital_dir), "exists": True, "images": image_count, "classes": class_names})
    return summary


def resolve_hospital_selection(hospital_datasets, hospital_paths, configured_count):
    if not hospital_datasets:
        return [], 0

    available_count = len(hospital_datasets)
    if hospital_paths:
        available_count = min(available_count, len(hospital_paths))

    selected_count = available_count
    if configured_count is not None and configured_count > 0 and configured_count < selected_count:
        selected_count = configured_count
        if hospital_paths:
            selected_count = available_count

    return hospital_datasets[:selected_count], selected_count


def validate_hospital_inputs(hospital_paths, uploaded_files):
    return bool(hospital_paths or uploaded_files)


def has_valid_training_history(history):
    if not isinstance(history, dict):
        return False
    for values in history.values():
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            arr = np.asarray(values)
            if arr.size > 0:
                return True
    return False


def _metric_last_value(history, key):
    if not isinstance(history, dict):
        return None
    values = history.get(key, [])
    arr = np.asarray(values, dtype="float32").reshape(-1)
    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        return None
    return float(finite_values[-1])


def build_local_metrics(history_by_client, client_labels=None, federated_round=1):
    rows = []
    for client_index, history in enumerate(history_by_client):
        if not isinstance(history, dict):
            continue
        client_name = client_labels[client_index] if client_labels and client_index < len(client_labels) else f"Hospital {client_index + 1}"
        epoch_count = max(
            [
                len(np.asarray(history.get(key, []), dtype="float32").reshape(-1))
                for key in ("loss", "accuracy", "val_loss", "val_accuracy")
                if key in history
            ]
            or [0]
        )
        for epoch in range(epoch_count):
            row = {
                "client": client_name,
                "federated_round": int(federated_round),
                "epoch": int(epoch + 1),
                "loss": _metric_last_value({key: np.asarray(history.get(key, []), dtype="float32").reshape(-1)[epoch:epoch + 1] for key in history if key in {"loss", "accuracy", "val_loss", "val_accuracy"}} , "loss"),
                "accuracy": _metric_last_value({key: np.asarray(history.get(key, []), dtype="float32").reshape(-1)[epoch:epoch + 1] for key in history if key in {"loss", "accuracy", "val_loss", "val_accuracy"}} , "accuracy"),
                "val_loss": _metric_last_value({key: np.asarray(history.get(key, []), dtype="float32").reshape(-1)[epoch:epoch + 1] for key in history if key in {"loss", "accuracy", "val_loss", "val_accuracy"}} , "val_loss"),
                "val_accuracy": _metric_last_value({key: np.asarray(history.get(key, []), dtype="float32").reshape(-1)[epoch:epoch + 1] for key in history if key in {"loss", "accuracy", "val_loss", "val_accuracy"}} , "val_accuracy"),
            }
            rows.append(row)
    return rows


def build_global_metrics(server_metrics, clients_count=0):
    rows = []
    for item in server_metrics or []:
        if not isinstance(item, dict):
            continue
        round_number = item.get("federated_round", item.get("round"))
        if round_number is None:
            continue
        rows.append({
            "federated_round": int(round_number),
            "round": int(round_number),
            "global_loss": item.get("loss", item.get("global_loss")),
            "global_accuracy": item.get("accuracy", item.get("global_accuracy")),
            "global_val_loss": item.get("val_loss", item.get("global_val_loss")),
            "global_val_accuracy": item.get("val_accuracy", item.get("global_val_accuracy")),
            "clients": item.get("clients", clients_count),
        })
    return rows


def split_train_val(images, labels, val_fraction=0.2, seed=42):
    if len(images) < 2:
        return (images, labels), (images[:1], labels[:1])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    split_index = max(1, int((1 - val_fraction) * len(images)))
    train_idx = indices[:split_index]
    val_idx = indices[split_index:]
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def run_training(images, labels, class_names, model_name, max_samples, epochs, batch_size):
    num_classes = len(class_names)
    train_data, val_data = split_train_val(images, labels)
    train_images, train_labels = train_data
    val_images, val_labels = val_data

    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_placeholder.info("Preparing training run...")

    log_panel = st.expander("Live run log", expanded=True)
    log_placeholder = log_panel.empty()
    _render_run_log(log_placeholder)

    client = build_client(model_name, num_classes)
    callbacks = [StreamlitTrainingCallback(status_placeholder, progress_bar, log_placeholder)]
    history = client.train_model(
        (train_images, train_labels),
        (val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    status_placeholder.success("Training completed. Evaluating model...")
    progress_bar.progress(1.0)
    accuracy = client.evaluate_model((val_images, val_labels))
    _append_run_log(f"Final evaluation accuracy: {accuracy:.4f}")
    return history, accuracy, client, (val_images, val_labels)


def run_federation(client, server, rounds=1, train_data=None, epochs=5, batch_size=32, callbacks=None, eval_data=None):
    if server.global_weights is None:
        server.global_weights = [np.array(weight, copy=True) for weight in client.model.get_weights()]

    if train_data is None:
        for _ in range(rounds):
            server.update_global_model(client.model.get_weights())
            server.apply_global_update(client)
            if eval_data is not None:
                server.evaluate_global_model([client], eval_data)
        return server

    for _ in range(rounds):
        client.model.set_weights(server.global_weights)
        train_images, train_labels = train_data
        history = client.train_model(
            (train_images, train_labels),
            (train_images, train_labels),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks or [],
        )
        client_weights = client.model.get_weights()
        aggregated_weights = server.aggregate_with_fedprox([client_weights], server.global_weights)
        server.update_global_model(aggregated_weights)
        client.model.set_weights(server.global_weights)
        if eval_data is not None:
            server.evaluate_global_model([client], eval_data)
    return server


def run_multi_client_federation(clients, server, rounds=1, client_datasets=None, epochs=5, batch_size=32, callbacks=None, eval_data=None):
    if not clients:
        return server

    if server.global_weights is None:
        server._initialize_global_weights(clients)

    if client_datasets is None:
        for _ in range(rounds):
            server.run_federated_round(clients, eval_data=eval_data)
        return server

    def train_client_round(client):
        dataset = client_datasets.get(client.client_id, client_datasets.get(client.client_id - 1, None))
        if dataset is None:
            dataset = client_datasets.get("default")
        if dataset is None:
            current_weights = client.model.get_weights()
            return current_weights, {"round": server.round + 1, "trained": False, "client_id": client.client_id}

        client_images, client_labels = dataset
        client.model.set_weights(server.global_weights)
        history = client.train_model(
            (client_images, client_labels),
            (client_images, client_labels),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks or [],
        )
        return client.model.get_weights(), history

    server.run_federated_training(clients, rounds=int(rounds), round_train_fn=train_client_round, eval_data=eval_data)
    return server


def explain_prediction(client, image, class_names, target_layer_name):
    """Compute Grad-CAM locally on the client model without sending raw medical images to the server."""
    gradcam = GradCAM(client.model, target_layer_name=target_layer_name)
    heatmap = gradcam.generate_heatmap(image, class_index=None)
    prediction = client.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    if prediction.ndim == 1 and len(prediction) == 1:
        probability = float(prediction[0])
        predicted_class = class_names[1 if probability >= 0.5 else 0]
    else:
        predicted_class = class_names[int(np.argmax(prediction))]
    overlay = gradcam.overlay_heatmap(image, heatmap, alpha=0.55)
    return heatmap, overlay, predicted_class, prediction


def resolve_target_layer_name(target_layer_name):
    if target_layer_name is None:
        return None
    cleaned = str(target_layer_name).strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return cleaned


def select_explainability_samples(images, labels, class_names, max_samples=4):
    if len(images) <= max_samples:
        return list(range(len(images)))

    selected = []
    for class_idx in range(len(class_names)):
        class_mask = labels == class_idx
        if np.any(class_mask):
            selected.append(int(np.flatnonzero(class_mask)[0]))
        if len(selected) >= max_samples:
            break

    if len(selected) < max_samples:
        remaining = [idx for idx in range(len(images)) if idx not in selected]
        selected.extend(remaining[: max_samples - len(selected)])

    return selected[:max_samples]


def build_confusion_metrics(client, val_data, class_names):
    val_images, val_labels = val_data
    predictions = client.model.predict(val_images, verbose=0)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        pred_labels = (predictions[:, 0] >= 0.5).astype(int)
    else:
        pred_labels = np.argmax(predictions, axis=1)
    return pred_labels, val_labels


st.title("Explainable Federated Learning for Medical Imaging")
st.caption("Privacy-preserving workflow: local hospital clients train on their own data, a central server only receives model updates, and Grad-CAM is computed locally on the client-side model for explanation.")

with st.sidebar:
    st.header("Configuration")
    st.caption("Local hospital data and a central federated server are used together here.")
    with st.form("pipeline_config"):
        uploaded_files = st.file_uploader("Upload raw medical images", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
        hospital_dataset_paths = st.text_area(
            "Hospital dataset paths",
            value="",
            help="Enter one dataset path per line. Each path represents a separate hospital client with its own local dataset. This is the required input for multi-hospital federated learning.",
        )
        result_dir = st.text_input(
            "Result directory",
            value=str(DEFAULT_RESULT_DIR),
            help="Path where preprocessed arrays and output artifacts will be stored outside the project folder.",
        )
        model_name = st.selectbox("Model", options=sorted(SUPPORTED_MODELS))
        hospital_count = st.number_input("Number of hospitals", min_value=2, max_value=6, value=2, step=1)
        target_accuracy = st.slider("Target accuracy", min_value=0.50, max_value=0.99, value=0.80, step=0.01, help="Adjust the required minimum validation accuracy for model acceptance.")
        max_samples = st.number_input("Max samples", min_value=16, max_value=2048, value=256, step=16)
        epochs = st.number_input("Local Epochs", min_value=1, max_value=20, value=5, step=1)
        batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=2, step=1)
        chunk_size = st.number_input("Chunk size", min_value=16, max_value=2048, value=256, step=16, help="Process a limited number of images at a time for large datasets")
        target_layer_name = st.text_input(
            "Grad-CAM target layer",
            value="auto",
            help="Use a specific layer name or leave as auto to use the last convolutional layer automatically.",
        )
        federation_backend = st.selectbox(
            "Federation backend",
            options=["custom", "flower"],
            index=0,
            help="Use the custom federated implementation or the real Flower backend when it is available.",
        )
        aggregation_mode = st.selectbox("Federated aggregation", options=["fedavg", "fedprox"], index=0)
        proximal_mu = st.number_input("FedProx mu", min_value=0.0, max_value=1.0, value=0.01, step=0.01, help="Only used when FedProx is selected")
        federated_rounds = st.number_input("Federated rounds", min_value=1, max_value=5, value=1, step=1)
        run_button = st.form_submit_button("Run pipeline")

if run_button:
    dataset_path = None
    hospital_paths = parse_hospital_dataset_paths(hospital_dataset_paths)
    if not validate_hospital_inputs(hospital_paths, uploaded_files):
        st.error("Please provide at least one hospital dataset path or upload at least one image")
        st.stop()
    if len(hospital_paths) == 1 and hospital_count > 1:
        st.warning("Only one hospital dataset path was entered. To compare multiple hospitals, enter one dataset directory per line.")

    if hospital_paths:
        hospital_summary = summarize_hospital_datasets(hospital_paths, result_dir=result_dir)
        st.caption("Hospital dataset summary")
        summary_rows = []
        for item in hospital_summary:
            summary_rows.append({
                "Hospital": Path(item["path"]).name if item["path"] else "Unknown",
                "Path": item["path"],
                "Exists": item["exists"],
                "Images": item["images"],
                "Classes": ", ".join(item["classes"]) if item["classes"] else "-",
            })
        st.dataframe(summary_rows, use_container_width=True)

    _reset_run_logs()
    _append_run_log("Starting pipeline...")

    with st.spinner("Preprocessing hospital datasets..."):
        status_placeholder = st.empty()
        status_placeholder.info("Loading and preprocessing each hospital dataset...")
        _append_run_log("Preprocessing hospital datasets...")
        if hospital_paths:
            hospital_datasets = load_hospital_datasets(
                hospital_paths,
                batch_size=batch_size,
                chunk_size=chunk_size,
                result_dir=result_dir,
            )
            images = np.concatenate([dataset["images"] for dataset in hospital_datasets], axis=0)
            labels = np.concatenate([dataset["labels"] for dataset in hospital_datasets], axis=0)
            class_names = hospital_datasets[0]["class_names"]
        else:
            images, labels, class_names = load_dataset(
                uploaded_files,
                batch_size=batch_size,
                chunk_size=chunk_size,
                dataset_path=dataset_path,
                result_dir=result_dir,
            )
            hospital_datasets = [{"images": images, "labels": labels, "class_names": class_names}]
        _append_run_log("Preprocessing finished. Starting training...")
        status_placeholder.success("Preprocessing finished. Starting training...")

    if len(images) < 2:
        st.error("At least two images are needed for training")
        st.stop()

    train_data, val_data = split_train_val(images, labels, seed=7)
    train_images, train_labels = train_data

    if hospital_paths:
        selected_datasets, client_count = resolve_hospital_selection(hospital_datasets, hospital_paths, int(hospital_count))
    else:
        selected_datasets = [{"images": images, "labels": labels, "class_names": class_names}]
        client_count = 1

    clients = []
    for client_index, hospital_dataset in enumerate(selected_datasets, start=1):
        client_images = hospital_dataset["images"]
        client_labels = hospital_dataset["labels"]
        if len(client_images) < 2:
            st.warning(f"Hospital {client_index} skipped: fewer than 2 valid images after preprocessing.")
            continue
        client = FederatedClient(
            client_id=client_index,
            server_address="local",
            model_name=model_name,
            num_classes=len(class_names),
        )
        clients.append(client)

    if not clients:
        st.error("No valid hospital clients were created. Check that each hospital dataset contains enough labeled images.")
        st.stop()

    st.success("Initialized global model and hospital clients for federated training")

    server = FederatedFlowerServer(model_name=model_name, aggregation_mode=aggregation_mode, mu=proximal_mu)
    with st.spinner("Running federated rounds: each hospital receives the global model, trains locally for the configured epochs, and sends updates for aggregation..."):
        if len(class_names) >= 2 and len(clients) >= 2:
            client_datasets = {
                client.client_id: (hospital_dataset["images"], hospital_dataset["labels"])
                for client, hospital_dataset in zip(clients, selected_datasets)
            }
            if federation_backend == "flower":
                if not is_flower_available():
                    st.warning("Flower is not installed in the active environment. Falling back to the custom federation backend.")
                    federation_backend = "custom"
                else:
                    server = run_flower_federation(
                        clients,
                        server,
                        rounds=int(federated_rounds),
                        client_datasets=client_datasets,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        eval_data=(val_data[0], val_data[1]),
                        aggregation_mode=aggregation_mode,
                        mu=proximal_mu,
                    )
                    server = server
            if federation_backend == "custom":
                server = run_multi_client_federation(
                    clients,
                    server,
                    rounds=int(federated_rounds),
                    client_datasets=client_datasets,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    eval_data=(val_data[0], val_data[1]),
                )
        else:
            if federation_backend == "flower":
                if not is_flower_available():
                    st.warning("Flower is not installed in the active environment. Falling back to the custom federation backend.")
                    federation_backend = "custom"
                else:
                    server = run_flower_federation(
                        clients,
                        server,
                        rounds=int(federated_rounds),
                        client_datasets={"default": (train_images, train_labels)},
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        eval_data=(val_data[0], val_data[1]),
                        aggregation_mode=aggregation_mode,
                        mu=proximal_mu,
                    )
            if federation_backend == "custom":
                server = run_federation(
                    clients[0],
                    server,
                    rounds=int(federated_rounds),
                    train_data=(train_images, train_labels),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    eval_data=(val_data[0], val_data[1]),
                )

    aggregated_client = clients[0]
    if server.global_weights is not None:
        aggregated_client.model.set_weights(server.global_weights)

    history_by_client = []
    for round_state in getattr(server, "client_histories", []):
        local_histories = round_state.get("local_histories", []) if isinstance(round_state, dict) else []
        for local_history in local_histories:
            if isinstance(local_history, dict) and has_valid_training_history(local_history):
                history_by_client.append(local_history)

    if not history_by_client:
        history_by_client = []

    history = aggregate_training_history(history_by_client)
    accuracy = aggregated_client.evaluate_model(val_data)

    accuracy_status = "Met" if accuracy >= float(target_accuracy) else "Below target"
    if accuracy >= float(target_accuracy):
        st.success(f"Model accuracy target achieved: {accuracy:.4f} >= {target_accuracy:.2f}")
    else:
        st.warning(f"Model accuracy is below the target: {accuracy:.4f} < {target_accuracy:.2f}. Increase epochs or improve preprocessing.")

    metrics_db_path = Path(result_dir).expanduser() / "training_metrics.db"
    local_metrics = build_local_metrics(history_by_client, [f"Hospital {index}" for index in range(1, len(history_by_client) + 1)], federated_round=int(federated_rounds))
    global_metrics = build_global_metrics(server.global_round_metrics, clients_count=int(len(clients)))
    run_id = save_run_metrics(
        db_path=metrics_db_path,
        model_name=model_name,
        aggregation_mode=aggregation_mode,
        config={
            "epochs": int(max(1, epochs)),
            "batch_size": int(batch_size),
            "hospital_count": int(len(clients)),
            "federated_rounds": int(federated_rounds),
            "class_names": [str(name) for name in class_names],
        },
        history_by_client=history_by_client,
        aggregated_history=history,
        local_metrics=local_metrics,
        global_metrics=global_metrics,
        accuracy=float(accuracy),
        val_accuracy=float(accuracy),
    )
    _append_run_log(f"Saved training metrics to {metrics_db_path} (run_id={run_id})")

    st.success("Federated aggregation completed")

    st.subheader("MODEL CONFIGURATION")
    config_values = [
        ("Model", model_name),
        ("Backend", federation_backend.upper()),
        ("Hospital Clients", int(len(clients))),
        ("Local Epochs", int(epochs)),
        ("Federated Rounds", int(federated_rounds)),
        ("Batch Size", int(batch_size)),
        ("Aggregation", aggregation_mode.upper()),
    ]
    config_cols = st.columns(len(config_values))
    for idx, (label, value) in enumerate(config_values):
        with config_cols[idx]:
            st.metric(label, str(value))

    st.markdown("## SECTION A: LOCAL CLIENT TRAINING")
    valid_histories = [entry for entry in history_by_client if has_valid_training_history(entry)]
    if valid_histories:
        client_labels = [f"Hospital {index}" for index in range(1, len(valid_histories) + 1)]
        for idx, history in enumerate(valid_histories):
            st.caption(client_labels[idx])
            metric_cols = st.columns(4)
            metric_keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
            for col_idx, key in enumerate(metric_keys):
                values = history.get(key, [])
                value = float(np.asarray(values).reshape(-1)[-1]) if isinstance(values, (list, tuple, np.ndarray)) and np.asarray(values).size > 0 else None
                label = {
                    "loss": "Training Loss",
                    "val_loss": "Validation Loss",
                    "accuracy": "Training Accuracy",
                    "val_accuracy": "Validation Accuracy",
                }.get(key, key.replace("_", " ").title())
                with metric_cols[col_idx]:
                    st.metric(label, f"{value:.4f}" if value is not None else "N/A")
        client_fig = plot_training_history_by_client(valid_histories, client_labels)
        st.pyplot(client_fig)
    else:
        st.info("Local client training history will appear here only after at least one hospital produces a valid local training history.")

    st.markdown("## SECTION B: FEDERATED GLOBAL TRAINING")
    if server.global_round_metrics:
        round_metrics_df = pd.DataFrame(server.global_round_metrics)
        if "round" not in round_metrics_df.columns and "federated_round" in round_metrics_df.columns:
            round_metrics_df = round_metrics_df.rename(columns={"federated_round": "round"})
        if "round" in round_metrics_df.columns:
            round_metrics_df = round_metrics_df.sort_values("round").reset_index(drop=True)
        round_metrics_df = round_metrics_df.copy()

        for existing, target in {
            "round": "Round",
            "loss": "Global Evaluation Loss",
            "accuracy": "Global Evaluation Accuracy",
            "val_loss": "Global Validation Loss",
            "val_accuracy": "Global Validation Accuracy",
            "clients": "Clients",
        }.items():
            if existing in round_metrics_df.columns:
                round_metrics_df = round_metrics_df.rename(columns={existing: target})

        if "Round" not in round_metrics_df.columns:
            round_metrics_df["Round"] = np.arange(1, len(round_metrics_df) + 1)

        table_columns = ["Round", "Global Evaluation Loss", "Global Evaluation Accuracy", "Clients"]
        for column_name in ["Global Validation Loss", "Global Validation Accuracy"]:
            if column_name not in round_metrics_df.columns:
                round_metrics_df[column_name] = np.nan
            if round_metrics_df[column_name].notna().any():
                table_columns.append(column_name)

        table_df = round_metrics_df[table_columns].copy()
        st.dataframe(table_df, use_container_width=True)

        if len(table_df) <= 1:
            st.info("Only one federated round has been completed. Increase the number of federated rounds to view a training curve.")
        else:
            st.subheader("1. Global Evaluation Accuracy vs Federated Round")
            accuracy_chart = table_df[["Round", "Global Evaluation Accuracy"]].dropna().set_index("Round")
            st.line_chart(accuracy_chart)

            st.subheader("2. Global Evaluation Loss vs Federated Round")
            loss_chart = table_df[["Round", "Global Evaluation Loss"]].dropna().set_index("Round")
            st.line_chart(loss_chart)

            if "Global Validation Accuracy" in table_df.columns and not table_df["Global Validation Accuracy"].isna().all():
                st.subheader("3. Global Validation Accuracy vs Federated Round")
                val_accuracy_chart = table_df[["Round", "Global Validation Accuracy"]].dropna().set_index("Round")
                st.line_chart(val_accuracy_chart)

            if "Global Validation Loss" in table_df.columns and not table_df["Global Validation Loss"].isna().all():
                st.subheader("4. Global Validation Loss vs Federated Round")
                val_loss_chart = table_df[["Round", "Global Validation Loss"]].dropna().set_index("Round")
                st.line_chart(val_loss_chart)
    else:
        st.info("Federated global training history will appear after the first aggregation step.")

    st.subheader("FINAL GLOBAL MODEL PERFORMANCE")
    pred_labels, true_labels = build_confusion_metrics(aggregated_client, val_data, class_names)
    metrics = compute_classification_metrics(true_labels, pred_labels, list(class_names))
    metric_cols = st.columns(3)
    display_items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1-score", metrics.get("f1_score")),
        ("Sensitivity", metrics.get("sensitivity")),
        ("Specificity", metrics.get("specificity")),
    ]
    for idx, (name, value) in enumerate(display_items):
        with metric_cols[idx % 3]:
            if value is None:
                st.metric(name, "Unavailable")
            else:
                st.metric(name, f"{float(value):.4f}")

    confusion_fig = plot_confusion_matrix(true_labels, pred_labels, list(class_names))
    st.pyplot(confusion_fig)

    st.subheader("EXPLAINABLE AI (LOCAL XAI)")
    st.caption("Medical Image → Local Trained CNN → Prediction → Local Grad-CAM → Heatmap/Explanation. Raw images are never transmitted to the federated server.")
    st.subheader("Recent metric runs")
    available_run_ids = list_run_ids(db_path=metrics_db_path, limit=20)
    current_run_id = st.session_state.get("current_run_id") or (available_run_ids[0] if available_run_ids else None)
    if available_run_ids:
        selected_run_id = st.selectbox(
            "Metric run selector",
            options=available_run_ids,
            index=available_run_ids.index(current_run_id) if current_run_id in available_run_ids else 0,
            help="Only metrics for the selected run are displayed. This prevents mixing different training runs.",
        )
        st.session_state["current_run_id"] = selected_run_id
    else:
        selected_run_id = current_run_id

    selected_run_metrics = load_run_metrics_by_id(db_path=metrics_db_path, run_id=selected_run_id)
    selected_local_metrics = load_run_local_metrics(db_path=metrics_db_path, run_id=selected_run_id)
    selected_global_metrics = load_run_global_metrics(db_path=metrics_db_path, run_id=selected_run_id)

    if selected_run_metrics:
        st.caption(f"Active run: {selected_run_id}")
        st.code(
            f"run_id={selected_run_id}\nmodel={selected_run_metrics.get('model_name')}\naggregation={selected_run_metrics.get('aggregation_mode')}\naccuracy={selected_run_metrics.get('accuracy')}\nval_accuracy={selected_run_metrics.get('val_accuracy')}",
            language="text",
        )

    if selected_local_metrics:
        local_run_df = pd.DataFrame(selected_local_metrics)
        st.dataframe(local_run_df, use_container_width=True)
    else:
        st.info("No local client metrics are stored for the selected run.")

    if selected_global_metrics:
        global_run_df = pd.DataFrame(selected_global_metrics)
        st.dataframe(global_run_df, use_container_width=True)
    else:
        st.info("No global federated metrics are stored for the selected run.")

    recent_runs = load_recent_metrics(db_path=metrics_db_path, limit=5)
    if recent_runs:
        recent_rows = []
        for row in recent_runs:
            recent_rows.append(
                {
                    "Run": row.get("run_id") or row["id"],
                    "Time": row["created_at"],
                    "Model": row["model_name"],
                    "Aggregation": row["aggregation_mode"],
                    "Accuracy": row["accuracy"],
                }
            )
        st_dataframe_compat(recent_rows, use_container_width=True)
    else:
        st.info("No prior run metrics were stored yet.")

    st.subheader("Explainability (Grad-CAM, computed locally on the client model)")
    resolved_target_layer = resolve_target_layer_name(target_layer_name)
    sample_indices = select_explainability_samples(images, labels, class_names, max_samples=4)

    st.caption("A small sample gallery shows how the model attends to different images and classes.")
    for sample_index in sample_indices:
        preview_image = images[sample_index]
        true_label = int(labels[sample_index]) if labels is not None and sample_index < len(labels) else None
        heatmap, overlay, predicted_class, prediction = explain_prediction(
            aggregated_client,
            preview_image,
            class_names,
            resolved_target_layer,
        )

        preview_image = np.asarray(preview_image).astype("float32")
        if preview_image.max() > 1.0 or preview_image.min() < 0.0:
            preview_image = (preview_image - preview_image.min()) / max(preview_image.max() - preview_image.min(), 1e-8)

        heatmap = np.asarray(heatmap).astype("float32")
        if heatmap.max() > 1.0 or heatmap.min() < 0.0:
            heatmap = (heatmap - heatmap.min()) / max(heatmap.max() - heatmap.min(), 1e-8)

        st.write(f"Sample {sample_index + 1}: predicted {predicted_class}" + (f" • true {class_names[true_label]}" if true_label is not None else ""))
        st.write(f"Prediction probabilities: {np.round(prediction, 3)}")

        image_col, overlay_col = st.columns(2)
        with image_col:
            st_image_compat(preview_image, caption="Input image", use_container_width=True)
        with overlay_col:
            st_image_compat(overlay, caption="Grad-CAM overlay", use_container_width=True)

        with st.expander("Raw Grad-CAM heatmap"):
            st_image_compat(heatmap, caption="Raw heatmap", use_container_width=True)

        st.divider()
