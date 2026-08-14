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
from metrics_store import DEFAULT_METRICS_DB, load_recent_metrics, save_run_metrics
from models.efficientnetb0 import FederatedClient, SUPPORTED_MODELS
from preprocessing.preprocess_images import ImagePreprocessor
from federated.server import FederatedFlowerServer, partition_dataset
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


def run_federation(client, server, rounds=1):
    if server.global_weights is None:
        server.update_global_model(client.model.get_weights())
    for _ in range(rounds):
        server.update_global_model(client.model.get_weights())
        server.apply_global_update(client)
    return server


def run_multi_client_federation(clients, server, rounds=1):
    for round_index in range(rounds):
        server.run_federated_round(clients)
        for client in clients:
            client.model.set_weights(server.global_weights)
    return server


def explain_prediction(client, image, class_names, target_layer_name):
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
st.caption("Run a privacy-preserving federated workflow where local hospital clients train on their own data, a Flower server aggregates updates, and Grad-CAM explains each prediction.")

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
        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=1, step=1)
        batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=2, step=1)
        chunk_size = st.number_input("Chunk size", min_value=16, max_value=2048, value=256, step=16, help="Process a limited number of images at a time for large datasets")
        target_layer_name = st.text_input(
            "Grad-CAM target layer",
            value="auto",
            help="Use a specific layer name or leave as auto to use the last convolutional layer automatically.",
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
    with st.spinner("Training hospital clients locally..."):
        clients = []
        history_by_client = []
        if hospital_paths:
            selected_datasets, client_count = resolve_hospital_selection(hospital_datasets, hospital_paths, int(hospital_count))
        else:
            selected_datasets = [{"images": images, "labels": labels, "class_names": class_names}]
            client_count = 1

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
            try:
                client_history = client.train_model(
                    (client_images, client_labels),
                    val_data,
                    epochs=max(1, epochs // 2),
                    batch_size=batch_size,
                    callbacks=[],
                )
            except Exception as exc:
                st.warning(f"Hospital {client_index} failed to train: {exc}")
                continue

            if not has_valid_training_history(client_history):
                st.warning(f"Hospital {client_index} produced no valid training history. Check image labeling and preprocessing.")
                continue

            history_by_client.append(client_history)
            clients.append(client)

    if not history_by_client or not clients:
        st.error("No hospital produced valid training history. Check that each hospital folder has labeled class directories and enough images.")
        st.stop()

    st.success("Local hospital training complete")

    server = FederatedFlowerServer(model_name=model_name, aggregation_mode=aggregation_mode, mu=proximal_mu)
    with st.spinner("Sharing model updates with the global server..."):
        if len(class_names) >= 2 and len(clients) >= 2:
            server = run_multi_client_federation(clients, server, rounds=int(federated_rounds))
        else:
            server = run_federation(clients[0], server, rounds=int(federated_rounds))

    aggregated_client = clients[0]
    if server.global_weights is not None:
        aggregated_client.model.set_weights(server.global_weights)
    history = aggregate_training_history(history_by_client)
    accuracy = aggregated_client.evaluate_model(val_data)

    accuracy_status = "Met" if accuracy >= float(target_accuracy) else "Below target"
    if accuracy >= float(target_accuracy):
        st.success(f"Model accuracy target achieved: {accuracy:.4f} >= {target_accuracy:.2f}")
    else:
        st.warning(f"Model accuracy is below the target: {accuracy:.4f} < {target_accuracy:.2f}. Increase epochs or improve preprocessing.")

    metrics_db_path = Path(result_dir).expanduser() / "training_metrics.db"
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
        accuracy=float(accuracy),
        val_accuracy=float(accuracy),
    )
    _append_run_log(f"Saved training metrics to {metrics_db_path} (run_id={run_id})")

    st.success("Federated aggregation completed")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training history")
        valid_histories = [entry for entry in history_by_client if has_valid_training_history(entry)]
        if valid_histories:
            client_labels = [f"Hospital {index}" for index in range(1, len(valid_histories) + 1)]
            st.caption("Local training curves for each hospital client")
            client_fig = plot_training_history_by_client(valid_histories, client_labels)
            st.pyplot(client_fig)

            if isinstance(history, dict) and history:
                st.caption("Aggregated training curve")
                aggregated_fig = plot_training_history(history)
                st.pyplot(aggregated_fig)
                chart_df = pd.DataFrame({
                    key: np.asarray(values, dtype="float32").reshape(-1)
                    for key, values in history.items()
                    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0
                })
                if not chart_df.empty:
                    st.line_chart(chart_df)

            if isinstance(history, dict):
                metric_items = [
                    (key, history[key][-1] if history[key] else None)
                    for key in ["loss", "val_loss", "accuracy", "val_accuracy", "binary_accuracy", "val_binary_accuracy"]
                    if key in history
                ]
                if metric_items:
                    metric_cols = st.columns(min(4, len(metric_items)))
                    for col, (key, value) in zip(metric_cols, metric_items):
                        if value is not None:
                            col.metric(key, f"{float(value):.4f}")
        elif history and isinstance(history, dict):
            fig = plot_training_history(history)
            st.pyplot(fig)
            chart_df = pd.DataFrame({
                key: np.asarray(values, dtype="float32").reshape(-1)
                for key, values in history.items()
                if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0
            })
            if not chart_df.empty:
                st.line_chart(chart_df)
            if isinstance(history, dict):
                metric_items = [
                    (key, history[key][-1] if history[key] else None)
                    for key in ["loss", "val_loss", "accuracy", "val_accuracy", "binary_accuracy", "val_binary_accuracy"]
                    if key in history
                ]
                if metric_items:
                    metric_cols = st.columns(min(4, len(metric_items)))
                    for col, (key, value) in zip(metric_cols, metric_items):
                        if value is not None:
                            col.metric(key, f"{float(value):.4f}")
        else:
            st.info("Training history will appear here only after at least one hospital produces a valid local training history.")

    with col2:
        st.subheader("Model summary")
        st.write({"Model": model_name, "Accuracy": round(float(accuracy), 4), "Target": round(float(target_accuracy), 4), "Status": accuracy_status, "Classes": list(class_names)})

        pred_labels, true_labels = build_confusion_metrics(aggregated_client, val_data, class_names)
        metrics = compute_classification_metrics(true_labels, pred_labels, list(class_names))
        metric_cols = st.columns(3)
        display_items = [
            ("Accuracy", metrics.get("accuracy", 0.0)),
            ("Precision", metrics.get("precision", 0.0)),
            ("Recall", metrics.get("recall", 0.0)),
            ("F1-score", metrics.get("f1_score", 0.0)),
            ("Sensitivity", metrics.get("sensitivity", 0.0)),
            ("Specificity", metrics.get("specificity", 0.0)),
            ("Balanced acc", metrics.get("balanced_accuracy", 0.0)),
        ]
        for idx, (name, value) in enumerate(display_items):
            with metric_cols[idx % 3]:
                st.metric(name, f"{float(value):.4f}")

        confusion_fig = plot_confusion_matrix(true_labels, pred_labels, list(class_names))
        st.pyplot(confusion_fig)

    st.subheader("Recent metric runs")
    recent_runs = load_recent_metrics(db_path=metrics_db_path, limit=5)
    if recent_runs:
        recent_rows = []
        for row in recent_runs:
            recent_rows.append(
                {
                    "Run": row["id"],
                    "Time": row["created_at"],
                    "Model": row["model_name"],
                    "Aggregation": row["aggregation_mode"],
                    "Accuracy": row["accuracy"],
                }
            )
        st_dataframe_compat(recent_rows, use_container_width=True)
    else:
        st.info("No prior run metrics were stored yet.")

    st.subheader("Explainability (Grad-CAM)")
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
