import hashlib
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

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
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HOSPITAL_CLASS_NAME_TO_ID = {"healthy": 0, "parkinson": 1}
HOSPITAL_CLASS_NAMES = ["healthy", "parkinson"]


def resolve_model_checkpoint_path(result_dir, model_name):
    checkpoint_dir = Path(result_dir).expanduser() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{model_name}_global_state.npz"


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
    remapped = np.asarray(labels, dtype=np.int32).copy()
    class_name_list = list(class_names)
    for class_name, target_index in mapping.items():
        if class_name in class_name_list:
            original_index = class_name_list.index(class_name)
            remapped[remapped == original_index] = target_index
    return remapped.astype(np.int32)


def _list_supported_image_files(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def validate_hospital_dataset(hospital_path):
    hospital_dir = Path(hospital_path).expanduser().resolve()
    if not hospital_dir.exists():
        raise FileNotFoundError(f"Hospital path does not exist: {hospital_dir}")

    class_paths = {}
    for class_name in HOSPITAL_CLASS_NAMES:
        class_dir = hospital_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Hospital {hospital_dir.name} is missing the required directory: {class_name}/"
            )
        class_paths[class_name] = _list_supported_image_files(class_dir)
        if not class_paths[class_name]:
            raise ValueError(
                f"Hospital {hospital_dir.name} contains no {class_name} images. Expected directory: {class_dir}"
            )

    healthy_paths = class_paths["healthy"]
    parkinson_paths = class_paths["parkinson"]
    all_paths = healthy_paths + parkinson_paths
    labels = np.array([0] * len(healthy_paths) + [1] * len(parkinson_paths), dtype=np.int32)

    print(f"\nHospital {hospital_dir.name}")
    print(f"Healthy images: {len(healthy_paths)}")
    print(f"Parkinson images: {len(parkinson_paths)}")
    print(f"Total images: {len(all_paths)}")
    print(f"Healthy path: {hospital_dir / 'healthy'}")
    print(f"Parkinson path: {hospital_dir / 'parkinson'}")

    return {
        "hospital": hospital_dir.name,
        "path": str(hospital_dir),
        "healthy_count": len(healthy_paths),
        "parkinson_count": len(parkinson_paths),
        "total_images": len(all_paths),
        "class_paths": class_paths,
        "images": all_paths,
        "labels": labels,
        "class_names": ["healthy", "parkinson"],
    }


def split_hospital_train_validation(images, labels, val_fraction=0.2, random_state=42):
    labels = np.asarray(labels).ravel()
    if len(images) != len(labels):
        raise ValueError("Image count and label count must match for hospital splitting.")
    if len(images) == 0:
        raise ValueError("Hospital dataset is empty; cannot create training and validation sets.")

    unique_counts = np.unique(labels, return_counts=True)[1]
    min_count = int(unique_counts.min()) if unique_counts.size else 0
    if min_count == 0:
        raise ValueError("Each hospital must contain both healthy and parkinson samples before split validation.")

    if min_count >= 2:
        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=val_fraction,
            random_state=random_state,
            stratify=labels,
        )
    else:
        warnings = "Validation split cannot contain both classes for this hospital because one class has only one sample. Falling back to the safest possible split."
        print(warnings)
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(images))
        split_index = max(1, int(len(images) * (1.0 - val_fraction)))
        train_idx = indices[:split_index]
        val_idx = indices[split_index:]
        train_images = [images[idx] for idx in train_idx]
        val_images = [images[idx] for idx in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

    return train_images, train_labels, val_images, val_labels


def validate_class_distribution(train_images, train_labels, val_images, val_labels):
    train_labels = np.asarray(train_labels).ravel()
    val_labels = np.asarray(val_labels).ravel()
    train_counts = {int(class_id): int(np.sum(train_labels == class_id)) for class_id in np.unique(train_labels)}
    val_counts = {int(class_id): int(np.sum(val_labels == class_id)) for class_id in np.unique(val_labels)}

    summary = {
        "train_healthy": int(train_counts.get(0, 0)),
        "train_parkinson": int(train_counts.get(1, 0)),
        "train_total": int(len(train_images)),
        "val_healthy": int(val_counts.get(0, 0)),
        "val_parkinson": int(val_counts.get(1, 0)),
        "val_total": int(len(val_images)),
        "training_classes_present": sorted(train_counts.keys()),
        "validation_classes_present": sorted(val_counts.keys()),
    }

    if 0 not in train_counts:
        raise ValueError("Training split is missing healthy samples.")
    if 1 not in train_counts:
        raise ValueError("Training split is missing parkinson samples.")
    if 0 not in val_counts:
        raise ValueError("Validation split is missing healthy samples.")
    if 1 not in val_counts:
        raise ValueError("Validation split is missing parkinson samples.")

    return summary


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


def load_hospital_datasets(hospital_paths, target_size=(224, 224), batch_size=8, chunk_size=256, result_dir=None, max_samples=None, random_state=42):
    datasets = []
    for hospital_path in hospital_paths:
        hospital_summary = validate_hospital_dataset(hospital_path)
        image_paths = hospital_summary["images"]
        labels = hospital_summary["labels"]

        if len(image_paths) == 0:
            raise ValueError(f"Hospital {hospital_path} does not contain any supported images.")

        if max_samples is not None:
            max_samples = max(2, int(max_samples))
            healthy_paths = hospital_summary["class_paths"]["healthy"]
            parkinson_paths = hospital_summary["class_paths"]["parkinson"]
            healthy_cap = min(len(healthy_paths), max(1, max_samples // 2))
            parkinson_cap = min(len(parkinson_paths), max(1, max_samples // 2))
            if healthy_cap + parkinson_cap < max_samples:
                extra_slots = max_samples - (healthy_cap + parkinson_cap)
                healthy_cap += min(extra_slots, len(healthy_paths) - healthy_cap)
                parkinson_cap += min(max(0, extra_slots - max(0, len(healthy_paths) - healthy_cap)), len(parkinson_paths) - parkinson_cap)
            rng = np.random.default_rng(random_state)
            healthy_selected = rng.choice(healthy_paths, size=healthy_cap, replace=False).tolist() if healthy_paths else []
            parkinson_selected = rng.choice(parkinson_paths, size=parkinson_cap, replace=False).tolist() if parkinson_paths else []
            selected_paths = healthy_selected + parkinson_selected
            selected_labels = np.array([0] * len(healthy_selected) + [1] * len(parkinson_selected), dtype=np.int32)
            image_paths = selected_paths
            labels = selected_labels

        image_arrays = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size)
                    image_arrays.append(np.asarray(img, dtype=np.float32) / 255.0)
            except Exception as exc:
                raise RuntimeError(f"Failed to load image {image_path} for hospital {hospital_path}: {exc}") from exc

        if not image_arrays:
            raise ValueError(f"No readable images were found in hospital dataset at {hospital_path}.")

        images = np.stack(image_arrays, axis=0)
        train_images, train_labels, val_images, val_labels = split_hospital_train_validation(images, labels, val_fraction=0.2, random_state=42)
        validate_class_distribution(train_images, train_labels, val_images, val_labels)

        datasets.append({
            "path": hospital_path,
            "hospital": hospital_summary["hospital"],
            "images": images,
            "labels": labels,
            "class_names": hospital_summary["class_names"],
            "train": (train_images, train_labels),
            "val": (val_images, val_labels),
            "healthy_count": int(np.sum(labels == 0)),
            "parkinson_count": int(np.sum(labels == 1)),
            "total_images": int(len(labels)),
        })
    return datasets


def summarize_hospital_datasets(hospital_paths, result_dir=None):
    summary = []
    for hospital_path in hospital_paths:
        hospital_dir = Path(hospital_path).expanduser()
        if not hospital_dir.exists():
            summary.append({"path": hospital_path, "exists": False, "images": 0, "classes": [], "healthy": 0, "parkinson": 0})
            continue

        healthy_count = 0
        parkinson_count = 0
        classes = []
        for class_name in ["healthy", "parkinson"]:
            class_dir = hospital_dir / class_name
            if class_dir.exists():
                classes.append(class_name)
                count = sum(1 for _ in _list_supported_image_files(class_dir))
                if class_name == "healthy":
                    healthy_count = count
                else:
                    parkinson_count = count
        image_count = healthy_count + parkinson_count
        summary.append({
            "path": str(hospital_dir),
            "exists": True,
            "images": image_count,
            "classes": classes,
            "healthy": healthy_count,
            "parkinson": parkinson_count,
        })
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
    for item in history_by_client:
        if isinstance(item, dict) and "history" in item:
            history = item.get("history")
            client_id = item.get("client_id")
            round_number = item.get("federated_round", federated_round)
        else:
            history = item
            client_id = None
            round_number = federated_round

        if not isinstance(history, dict):
            continue

        client_name = client_labels[client_id] if isinstance(client_labels, dict) and client_id in client_labels else (
            client_id if client_id is not None else (
                client_labels[0] if isinstance(client_labels, list) and client_labels else f"Hospital {len(rows) + 1}"
            )
        )

        epoch_count = max(
            [
                len(np.asarray(history.get(key, []), dtype="float32").reshape(-1))
                for key in ("loss", "accuracy", "val_loss", "val_accuracy")
                if key in history
            ]
            or [0]
        )
        for epoch in range(epoch_count):
            metric_slice = {
                key: np.asarray(history.get(key, []), dtype="float32").reshape(-1)[epoch:epoch + 1]
                for key in history if key in {"loss", "accuracy", "val_loss", "val_accuracy"}
            }
            row = {
                "client": client_name,
                "federated_round": int(round_number),
                "epoch": int(epoch + 1),
                "loss": _metric_last_value(metric_slice, "loss"),
                "accuracy": _metric_last_value(metric_slice, "accuracy"),
                "val_loss": _metric_last_value(metric_slice, "val_loss"),
                "val_accuracy": _metric_last_value(metric_slice, "val_accuracy"),
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
            "validation_samples": item.get("local_validation_samples", item.get("validation_samples")),
            "hospital_metrics": item.get("hospital_metrics", []),
        })
    return rows


def split_train_val(images, labels, val_fraction=0.2, seed=42):
    images = np.asarray(images)
    labels = np.asarray(labels).ravel()

    if len(images) == 0:
        raise ValueError("Dataset is empty; cannot create train/validation split.")
    if len(images) != len(labels):
        raise ValueError("Image count and label count must match for the train/validation split.")
    if len(images) < 2:
        return (images, labels), (images[:1], labels[:1])

    unique_classes, counts = np.unique(labels, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError("Each hospital must contain both healthy and parkinson samples before the train/validation split.")

    if counts.min() >= 2:
        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=val_fraction,
            random_state=seed,
            stratify=labels,
        )
        return (train_images, train_labels), (val_images, val_labels)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    split_index = max(1, int(len(images) * (1.0 - val_fraction)))
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
        raise ValueError("train_data is required for the correct federated workflow: each round must train the client from the latest global model.")

    if eval_data is not None and not isinstance(eval_data, dict):
        validation_data = eval_data
    else:
        validation_data = eval_data.get(1, eval_data) if isinstance(eval_data, dict) else None

    for round_index in range(1, int(rounds) + 1):
        client.model.set_weights(server.global_weights)
        train_images, train_labels = train_data
        if validation_data is None:
            raise ValueError("A pre-split validation set is required for the correct federated workflow; do not split training data again inside the training loop.")
        val_images, val_labels = validation_data
        history = client.train_model(
            (train_images, train_labels),
            (val_images, val_labels),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks or [],
        )
        client_weights = client.model.get_weights()
        aggregated_weights = server.aggregate_with_fedprox([client_weights], server.global_weights, sample_counts=[len(train_images)])
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
        raise ValueError("client_datasets is required for the correct federated workflow: each hospital must train locally every round from the latest global model.")

    def train_client_round(client):
        dataset = client_datasets.get(client.client_id, client_datasets.get(client.client_id - 1, None))
        if dataset is None:
            dataset = client_datasets.get("default")
        if dataset is None:
            current_weights = client.model.get_weights()
            return current_weights, {"round": server.round + 1, "trained": False, "client_id": client.client_id}, 0

        if isinstance(dataset, dict):
            train_data = dataset.get("train") or dataset.get("train_data")
            val_data = dataset.get("val") or dataset.get("validation") or dataset.get("val_data")
            if train_data is None or val_data is None:
                raise ValueError(f"Client {client.client_id} is missing a pre-split train/validation pair. Do not split again inside the federated loop.")
        elif len(dataset) == 4:
            train_data = (dataset[0], dataset[1])
            val_data = (dataset[2], dataset[3])
        elif len(dataset) == 2:
            raise ValueError(
                f"Client {client.client_id} dataset is only a single split. "
                "Provide train_data and validation_data before federation; do not split again inside the server."
            )
        else:
            raise ValueError(f"Client {client.client_id} dataset is in an unsupported format: expected train/val pair or 4-tuple.")

        client.model.set_weights(server.global_weights)
        history = client.train_model(
            train_data,
            val_data,
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks or [],
        )
        sample_count = int(len(train_data[0]))
        return client.model.get_weights(), history, sample_count

    server.run_federated_training(clients, rounds=int(rounds), round_train_fn=train_client_round, eval_data=eval_data)
    return server


def explain_prediction(client, image, class_names, target_layer_name):
    """Compute Grad-CAM locally on the client model without sending raw medical images to the server."""
    gradcam = GradCAM(client.model, target_layer_name=target_layer_name)
    heatmap = gradcam.generate_heatmap(image, class_index=None)
    prediction = client.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    if prediction.ndim == 1 and len(prediction) == 1:
        probability = float(prediction[0])
        class_names = list(class_names) if class_names else ["class_0", "class_1"]
        if len(class_names) < 2:
            class_names = ["class_0", "class_1"]
        predicted_class = class_names[1 if probability >= 0.5 else 0]
    else:
        class_names = list(class_names) if class_names else ["class_0", "class_1"]
        predicted_class = class_names[int(np.argmax(prediction))]
    overlay = gradcam.overlay_heatmap(image, heatmap, alpha=0.55)
    return heatmap, overlay, predicted_class, prediction


def summarize_prediction_probabilities(prediction, class_names):
    labels = list(class_names) if class_names else ["class_0", "class_1"]
    if len(labels) < 2:
        labels = ["class_0", "class_1"]

    prob_vector = np.asarray(prediction, dtype="float32").ravel()
    if prob_vector.size == 1:
        positive_prob = float(prob_vector[0])
        negative_prob = 1.0 - positive_prob
        return {
            labels[0]: float(negative_prob),
            labels[1]: float(positive_prob),
        }

    probs = [float(value) for value in prob_vector[: len(labels)]]
    if len(probs) < len(labels):
        probs = probs + [0.0] * (len(labels) - len(probs))
    return {label: float(prob) for label, prob in zip(labels, probs)}


def resolve_target_layer_name(target_layer_name):
    if target_layer_name is None:
        return None
    cleaned = str(target_layer_name).strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return cleaned


def select_explainability_samples(images, labels, class_names, max_samples=4):
    if images is None or labels is None:
        return []
    if not hasattr(images, "__len__"):
        return []
    if len(images) == 0:
        return []
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


def resolve_active_run_id(available_run_ids, current_run_id=None, preferred_run_id=None):
    available = list(available_run_ids or [])
    if preferred_run_id and preferred_run_id in available:
        return preferred_run_id
    if current_run_id and current_run_id in available:
        return current_run_id
    return available[0] if available else None


def build_confusion_metrics(client, val_data, class_names):
    val_images, val_labels = val_data
    predictions = client.model.predict(val_images, verbose=0)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        pred_labels = (predictions[:, 0] >= 0.5).astype(int)
    else:
        pred_labels = np.argmax(predictions, axis=1)
    return pred_labels, val_labels


def validate_train_validation_overlap(train_images, val_images):
    train_set = set()
    val_set = set()
    for image in np.asarray(train_images):
        image_bytes = np.ascontiguousarray(image).tobytes()
        image_signature = hashlib.sha256(
            image_bytes + str(np.asarray(image).shape).encode() + str(np.asarray(image).dtype).encode()
        ).hexdigest()
        train_set.add(image_signature)
    for image in np.asarray(val_images):
        image_bytes = np.ascontiguousarray(image).tobytes()
        image_signature = hashlib.sha256(
            image_bytes + str(np.asarray(image).shape).encode() + str(np.asarray(image).dtype).encode()
        ).hexdigest()
        val_set.add(image_signature)
    overlap = train_set.intersection(val_set)
    return bool(overlap), overlap


def summarize_validation_class_distribution(eval_images, eval_labels):
    labels = np.asarray(eval_labels).ravel()
    counts = {int(class_id): int(np.sum(labels == class_id)) for class_id in np.unique(labels)}
    return counts


def build_global_evaluation_diagnostics(client_eval_data):
    rows = []
    for client_id, val_data in sorted(client_eval_data.items(), key=lambda item: item[0]):
        if val_data is None:
            continue
        eval_images, eval_labels = val_data
        counts = summarize_validation_class_distribution(eval_images, eval_labels)
        rows.append({
            "Hospital": f"Hospital {client_id}",
            "Validation Samples": int(len(eval_images)),
            "Class 0": int(counts.get(0, 0)),
            "Class 1": int(counts.get(1, 0)),
        })
    return rows

st.title("Explainable Federated Learning for Medical Imaging")
st.caption("Privacy-preserving workflow: local hospital clients train on their own data, a central server only receives model updates, and Grad-CAM is computed locally on the client-side model for explanation.")

with st.sidebar:
    st.header("Configuration")
    st.caption("Local hospital data and a central federated server are used together here.")
    with st.form("pipeline_config"):
        uploaded_files = st.file_uploader("Upload raw medical images", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} uploaded image(s) detected. Click Run pipeline to preprocess and use them in training or Grad-CAM.")
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

display_images = st.session_state.get("processed_dataset", {}).get("images")
display_labels = st.session_state.get("processed_dataset", {}).get("labels")
display_class_names = st.session_state.get("processed_dataset", {}).get("class_names")

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
                max_samples=int(max_samples),
            )
            class_names = hospital_datasets[0]["class_names"]
            images = None
            labels = None
        else:
            images, labels, class_names = load_dataset(
                uploaded_files,
                batch_size=batch_size,
                chunk_size=chunk_size,
                dataset_path=dataset_path,
                result_dir=result_dir,
            )
            hospital_datasets = [{"images": images, "labels": labels, "class_names": class_names}]
        st.session_state["processed_dataset"] = {
            "images": images,
            "labels": labels,
            "class_names": class_names,
        }
        display_images = images
        display_labels = labels
        display_class_names = class_names
        _append_run_log("Preprocessing finished. Starting training...")
        status_placeholder.success("Preprocessing finished. Starting training...")

    if hospital_paths:
        selected_datasets, client_count = resolve_hospital_selection(hospital_datasets, hospital_paths, int(hospital_count))
    else:
        selected_datasets = [{"images": images, "labels": labels, "class_names": class_names}]
        client_count = 1

    client_datasets = {}
    client_eval_data = {}
    hospital_debug_summary = []
    for client_index, hospital_dataset in enumerate(selected_datasets, start=1):
        local_images = hospital_dataset["images"]
        local_labels = hospital_dataset["labels"]
        if len(local_images) < 2:
            continue
        local_train_data, local_val_data = split_train_val(local_images, local_labels, seed=7 + client_index)
        train_overlap, overlap_hashes = validate_train_validation_overlap(local_train_data[0], local_val_data[0])
        if train_overlap:
            st.error(f"DATA LEAKAGE DETECTED for Hospital {client_index}: {len(overlap_hashes)} duplicated image(s) across train and validation.")
            st.stop()
        train_counts = summarize_validation_class_distribution(local_train_data[0], local_train_data[1])
        val_counts = summarize_validation_class_distribution(local_val_data[0], local_val_data[1])
        if len(np.unique(local_val_data[1])) <= 1:
            st.warning(f"Validation set for Hospital {client_index} contains only one class; accuracy may be misleading.")
        hospital_debug_summary.append({
            "Hospital": f"Hospital {client_index}",
            "Train Samples": int(len(local_train_data[0])),
            "Validation Samples": int(len(local_val_data[0])),
            "Train Class 0": int(train_counts.get(0, 0)),
            "Train Class 1": int(train_counts.get(1, 0)),
            "Validation Class 0": int(val_counts.get(0, 0)),
            "Validation Class 1": int(val_counts.get(1, 0)),
        })
        client_datasets[client_index] = {"train": local_train_data, "val": local_val_data}
        client_eval_data[client_index] = local_val_data

    if hospital_debug_summary:
        st.dataframe(pd.DataFrame(hospital_debug_summary), use_container_width=True)

    if not hospital_paths and images is not None and labels is not None:
        if len(images) < 2:
            st.error("At least two images are needed for training")
            st.stop()
        train_data, val_data = split_train_val(images, labels, seed=7)
        train_images, train_labels = train_data
    elif hospital_paths and not client_datasets:
        st.error("No valid hospital datasets were available after preprocessing.")
        st.stop()

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
    checkpoint_path = resolve_model_checkpoint_path(result_dir, model_name)
    restored_state = server.load_state(checkpoint_path)
    if restored_state is not None:
        for client in clients:
            client.model.set_weights(server.global_weights)
        st.caption(f"Loaded previous federated model state from {checkpoint_path}")
    with st.spinner("Running federated rounds: each hospital receives the global model, trains locally for the configured epochs, and sends updates for aggregation..."):
        if len(class_names) >= 2 and len(clients) >= 2:
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
                        eval_data=client_eval_data,
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
                    eval_data=client_eval_data,
                )
        else:
            single_client_eval = client_eval_data.get(1, (train_images, train_labels))
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
                        eval_data={1: single_client_eval} if len(clients) == 1 else single_client_eval,
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
                    eval_data={1: single_client_eval} if len(clients) == 1 else single_client_eval,
                )

    if server.global_weights is not None:
        server.save_state(checkpoint_path)
        _append_run_log(f"Saved federated model checkpoint to {checkpoint_path}")

    aggregated_client = clients[0]
    if server.global_weights is not None:
        aggregated_client.model.set_weights(server.global_weights)

    history_by_client = []
    for round_state in getattr(server, "client_histories", []):
        local_histories = round_state.get("local_histories", []) if isinstance(round_state, dict) else []
        for item in local_histories:
            if isinstance(item, dict):
                local_history = item.get("history", item)
                if has_valid_training_history(local_history):
                    history_by_client.append({
                        "client_id": item.get("client_id", "unknown"),
                        "federated_round": item.get("federated_round", round_state.get("round")),
                        "history": local_history,
                    })

    history = aggregate_training_history([
        item.get("history", item) for item in history_by_client if isinstance(item, dict)
    ])
    final_global_eval = server.evaluate_global_model(clients, client_eval_data) if client_eval_data else None
    accuracy = float(final_global_eval["accuracy"]) if isinstance(final_global_eval, dict) and "accuracy" in final_global_eval else 0.0

    accuracy_status = "Met" if accuracy >= float(target_accuracy) else "Below target"
    if accuracy >= float(target_accuracy):
        st.success(f"Model accuracy target achieved: {accuracy:.4f} >= {target_accuracy:.2f}")
    else:
        st.warning(f"Model accuracy is below the target: {accuracy:.4f} < {target_accuracy:.2f}. Increase epochs or improve preprocessing.")

    metrics_db_path = Path(result_dir).expanduser() / "training_metrics.db"
    client_name_lookup = {
        item.get("client_id"): f"Hospital {item.get('client_id')}"
        for item in history_by_client
        if isinstance(item, dict) and item.get("client_id") is not None
    }
    local_metrics = build_local_metrics(history_by_client, client_name_lookup, federated_round=int(federated_rounds))
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
        val_accuracy=None,
        val_loss=None,
    )
    st.session_state["current_run_id"] = run_id
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
    valid_history_entries = [entry for entry in history_by_client if isinstance(entry, dict) and has_valid_training_history(entry.get("history", entry))]
    if valid_history_entries:
        grouped_by_hospital = {}
        for entry in valid_history_entries:
            hospital_id = entry.get("client_id", "unknown")
            round_number = entry.get("federated_round", 1)
            grouped_by_hospital.setdefault(f"Hospital {hospital_id}", {})[round_number] = entry.get("history", entry)

        hospital_names = sorted(grouped_by_hospital, key=lambda name: str(name))
        with st.form("hospital_round_view"):
            selected_hospital = st.selectbox(
                "Select Hospital",
                options=hospital_names,
                key="selected_hospital_name",
            )
            selected_round = st.selectbox(
                "Select Federated Round",
                options=sorted(grouped_by_hospital[selected_hospital]),
                key="selected_round_number",
            )
            st.form_submit_button("Apply view")
        history = grouped_by_hospital[selected_hospital][selected_round]
        st.subheader(f"{selected_hospital} • Round {selected_round}")

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

        st.caption("X-axis = Epoch")
        st.pyplot(plot_training_history(history))
    else:
        st.info("Local client training history will appear here only after at least one hospital produces a valid local training history.")

    st.markdown("## SECTION B: GLOBAL FEDERATED TRAINING")
    if server.global_round_metrics:
        round_metrics_df = pd.DataFrame(server.global_round_metrics)
        if "round" not in round_metrics_df.columns and "federated_round" in round_metrics_df.columns:
            round_metrics_df = round_metrics_df.rename(columns={"federated_round": "round"})
        if "round" in round_metrics_df.columns:
            round_metrics_df = round_metrics_df.sort_values("round").reset_index(drop=True)
        round_metrics_df = round_metrics_df.copy()

        for existing, target in {
            "round": "Round",
            "loss": "Loss",
            "accuracy": "Accuracy",
            "val_loss": "Validation Loss",
            "val_accuracy": "Validation Accuracy",
            "clients": "Clients",
            "validation_samples": "Validation Samples",
            "local_validation_samples": "Validation Samples",
        }.items():
            if existing in round_metrics_df.columns:
                round_metrics_df = round_metrics_df.rename(columns={existing: target})

        if "Round" not in round_metrics_df.columns:
            round_metrics_df["Round"] = np.arange(1, len(round_metrics_df) + 1)
        if "Validation Samples" not in round_metrics_df.columns:
            round_metrics_df["Validation Samples"] = np.nan

        validation_columns = ["Validation Loss", "Validation Accuracy"]
        for column_name in validation_columns:
            if column_name not in round_metrics_df.columns:
                round_metrics_df[column_name] = np.nan

        table_columns = ["Round", "Loss", "Accuracy", "Clients", "Validation Samples"]
        available_validation = [
            column_name for column_name in validation_columns
            if round_metrics_df[column_name].notna().any()
        ]
        if available_validation:
            table_columns.extend(available_validation)
        else:
            st.info("Global validation metrics unavailable for this run.")

        table_df = round_metrics_df[table_columns].copy()
        st.dataframe(table_df, use_container_width=True)

        if len(table_df) <= 1:
            st.info("Only one federated round has been completed. Increase the number of federated rounds to view a training curve.")
        else:
            round_values = table_df["Round"].astype(int).to_numpy()

            st.subheader("Global Evaluation Loss vs Federated Round")
            loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
            loss_ax.plot(round_values, table_df["Loss"].to_numpy(), marker="o")
            loss_ax.set_xticks(round_values)
            loss_ax.set_xlabel("Federated Round")
            loss_ax.set_ylabel("Loss")
            loss_ax.set_title("Global Evaluation Loss vs Federated Round")
            scr_loss = loss_fig.tight_layout()
            st.pyplot(loss_fig)

            st.subheader("Global Evaluation Accuracy vs Federated Round")
            acc_fig, acc_ax = plt.subplots(figsize=(6, 4))
            acc_ax.plot(round_values, table_df["Accuracy"].to_numpy(), marker="o")
            acc_ax.set_xticks(round_values)
            acc_ax.set_xlabel("Federated Round")
            acc_ax.set_ylabel("Accuracy")
            acc_ax.set_title("Global Evaluation Accuracy vs Federated Round")
            acc_fig.tight_layout()
            st.pyplot(acc_fig)

            if "Validation Accuracy" in table_df.columns and not table_df["Validation Accuracy"].isna().all():
                st.subheader("Global Validation Accuracy vs Federated Round")
                val_acc_fig, val_acc_ax = plt.subplots(figsize=(6, 4))
                val_acc_ax.plot(round_values, table_df["Validation Accuracy"].to_numpy(), marker="o")
                val_acc_ax.set_xticks(round_values)
                val_acc_ax.set_xlabel("Federated Round")
                val_acc_ax.set_ylabel("Validation Accuracy")
                val_acc_ax.set_title("Global Validation Accuracy vs Federated Round")
                val_acc_fig.tight_layout()
                st.pyplot(val_acc_fig)

            if "Validation Loss" in table_df.columns and not table_df["Validation Loss"].isna().all():
                st.subheader("Global Validation Loss vs Federated Round")
                val_loss_fig, val_loss_ax = plt.subplots(figsize=(6, 4))
                val_loss_ax.plot(round_values, table_df["Validation Loss"].to_numpy(), marker="o")
                val_loss_ax.set_xticks(round_values)
                val_loss_ax.set_xlabel("Federated Round")
                val_loss_ax.set_ylabel("Validation Loss")
                val_loss_ax.set_title("Global Validation Loss vs Federated Round")
                val_loss_fig.tight_layout()
                st.pyplot(val_loss_fig)
    else:
        st.info("Federated global training history will appear after the first aggregation step.")

    st.subheader("FINAL GLOBAL MODEL PERFORMANCE")
    if len(clients) > 1:
        combined_true = []
        combined_pred = []
        for client_id, val_data in sorted(client_eval_data.items()):
            if val_data is None:
                continue
            client_model = next((client for client in clients if getattr(client, "client_id", None) == client_id), None)
            if client_model is None:
                continue
            client_model.model.set_weights(server.global_weights)
            local_pred_labels, local_true_labels = build_confusion_metrics(client_model, val_data, class_names)
            combined_true.extend(np.asarray(local_true_labels).ravel().tolist())
            combined_pred.extend(np.asarray(local_pred_labels).ravel().tolist())
        if combined_true and combined_pred:
            metrics = compute_classification_metrics(np.asarray(combined_true), np.asarray(combined_pred), list(class_names))
        else:
            metrics = {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "sensitivity": None, "specificity": None}
    else:
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

    if len(clients) > 1 and combined_true and combined_pred:
        confusion_fig = plot_confusion_matrix(np.asarray(combined_true), np.asarray(combined_pred), list(class_names))
    else:
        confusion_fig = plot_confusion_matrix(true_labels, pred_labels, list(class_names))
    st.pyplot(confusion_fig)

    st.subheader("EXPLAINABLE AI (LOCAL XAI)")
    st.caption("Medical Image → Local Trained CNN → Prediction → Local Grad-CAM → Heatmap/Explanation. Raw images are never transmitted to the federated server.")
    st.subheader("Recent metric runs")
    available_run_ids = list_run_ids(db_path=metrics_db_path, limit=20)
    current_run_id = resolve_active_run_id(
        available_run_ids,
        current_run_id=st.session_state.get("current_run_id"),
        preferred_run_id=(run_id if "run_id" in locals() else None),
    )
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
    if display_images is None and uploaded_files:
        try:
            display_images, display_labels, display_class_names = load_dataset(
                uploaded_files,
                batch_size=batch_size,
                chunk_size=chunk_size,
                dataset_path=dataset_path,
                result_dir=result_dir,
            )
            st.session_state["processed_dataset"] = {
                "images": display_images,
                "labels": display_labels,
                "class_names": display_class_names,
            }
        except Exception as exc:
            st.warning(f"Uploaded image preprocessing failed for Grad-CAM preview: {exc}")
    resolved_target_layer = resolve_target_layer_name(target_layer_name)
    sample_indices = select_explainability_samples(display_images, display_labels, display_class_names, max_samples=4)

    if not sample_indices:
        st.info("No valid images were loaded for explainability preview. Upload images or run preprocessing to enable Grad-CAM visualizations.")
    else:
        st.caption("A small sample gallery shows how the model attends to different images and classes.")
        for sample_index in sample_indices:
            preview_image = display_images[sample_index]
            true_label = int(display_labels[sample_index]) if display_labels is not None and sample_index < len(display_labels) else None
            heatmap, overlay, predicted_class, prediction = explain_prediction(
                aggregated_client,
                preview_image,
                display_class_names,
                resolved_target_layer,
            )

            preview_image = np.asarray(preview_image).astype("float32")
            if preview_image.max() > 1.0 or preview_image.min() < 0.0:
                preview_image = (preview_image - preview_image.min()) / max(preview_image.max() - preview_image.min(), 1e-8)

            heatmap = np.asarray(heatmap).astype("float32")
            if heatmap.max() > 1.0 or heatmap.min() < 0.0:
                heatmap = (heatmap - heatmap.min()) / max(heatmap.max() - heatmap.min(), 1e-8)

            probability_map = summarize_prediction_probabilities(prediction, display_class_names)
            st.write(f"Sample {sample_index + 1}: predicted {predicted_class}" + (f" • true {display_class_names[true_label]}" if true_label is not None and true_label < len(display_class_names) else ""))
            st.write(
                "Prediction probabilities: "
                + ", ".join(f"{label}={value:.3f}" for label, value in probability_map.items())
            )

            image_col, overlay_col = st.columns(2)
            with image_col:
                st_image_compat(preview_image, caption="Input image", use_container_width=True)
            with overlay_col:
                st_image_compat(overlay, caption="Grad-CAM overlay", use_container_width=True)

            with st.expander("Raw Grad-CAM heatmap"):
                st_image_compat(heatmap, caption="Raw heatmap", use_container_width=True)

            st.divider()
