import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def aggregate_training_history(history_list):
    if not history_list:
        return {}

    histories = []
    for history in history_list:
        if not history:
            continue
        if isinstance(history, dict):
            metric_series = {
                key: values
                for key, values in history.items()
                if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0
            }
            if metric_series:
                histories.append(metric_series)

    if not histories:
        return {}

    all_keys = sorted({key for history in histories for key in history})
    aggregated = {}
    for key in all_keys:
        values = [np.asarray(history[key], dtype="float32") for history in histories if key in history]
        if not values:
            continue
        max_length = max(len(v) for v in values)
        padded = []
        for item in values:
            if len(item) < max_length:
                padded.append(np.pad(item, (0, max_length - len(item)), mode="constant", constant_values=np.nan))
            else:
                padded.append(item[:max_length])
        stacked = np.stack(padded, axis=0)
        aggregated[key] = np.nanmean(stacked, axis=0)

    return aggregated


def _coerce_metric_list(values):
    if values is None:
        return []
    if isinstance(values, (list, tuple, np.ndarray)):
        flattened = np.asarray(values, dtype="float32")
        return flattened.reshape(-1).tolist()
    if np.isscalar(values):
        return [float(values)]
    return []


def _extract_metric_series(history):
    if isinstance(history, dict):
        series = {}
        for key, values in history.items():
            if key.startswith("_"):
                continue
            coerced = _coerce_metric_list(values)
            if coerced:
                series[key] = coerced
        return series
    return {}


def _preferred_metric_keys():
    return [
        "loss",
        "val_loss",
        "accuracy",
        "val_accuracy",
        "binary_accuracy",
        "val_binary_accuracy",
        "sparse_categorical_accuracy",
        "val_sparse_categorical_accuracy",
    ]


def _pick_plot_metrics(metric_series):
    preferred = [key for key in _preferred_metric_keys() if key in metric_series]
    if preferred:
        return preferred
    return list(metric_series.keys())[:4]


def plot_training_history_by_client(histories, labels=None):
    if not histories:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(len(histories), 1, figsize=(7, 2.8 * len(histories)), squeeze=False)
    axes = axes.flatten()

    for idx, history in enumerate(histories):
        metric_series = _extract_metric_series(history)
        if not metric_series:
            axes[idx].text(0.5, 0.5, "No numeric metrics available for this client", ha="center", va="center")
            axes[idx].set_axis_off()
            continue

        selected_keys = _pick_plot_metrics(metric_series)
        plotted_any = False
        max_epoch_count = 0

        for key in selected_keys:
            values = np.asarray(metric_series[key], dtype="float32").reshape(-1)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            max_epoch_count = max(max_epoch_count, len(values))
            axes[idx].plot(np.arange(1, len(values) + 1), values, label=key, linewidth=1.8)
            plotted_any = True

        if not plotted_any:
            axes[idx].text(0.5, 0.5, "No numeric metrics available for this client", ha="center", va="center")
            axes[idx].set_axis_off()
            continue

        if max_epoch_count > 0:
            axes[idx].set_xticks(np.arange(1, max_epoch_count + 1))

        client_label = labels[idx] if labels and idx < len(labels) else f"Client {idx + 1}"
        axes[idx].set_title(f"Local Client: {client_label}")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Metric")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc="best")

    fig.tight_layout()
    return fig


def plot_training_history(history):
    if not history:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    if isinstance(history, list):
        metric_series = aggregate_training_history(history)
        round_values = None
    else:
        round_values = None
        if "round" in history:
            round_values = np.asarray(history["round"], dtype="float32").reshape(-1)
        elif "federated_round" in history:
            round_values = np.asarray(history["federated_round"], dtype="float32").reshape(-1)
        metric_series = _extract_metric_series(history)
        if "round" in metric_series:
            del metric_series["round"]
        if "federated_round" in metric_series:
            del metric_series["federated_round"]

    if not metric_series:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    selected_keys = _pick_plot_metrics(metric_series)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in selected_keys:
        values = np.asarray(metric_series[key], dtype="float32").reshape(-1)
        if values.size:
            if round_values is not None and len(round_values) >= len(values):
                x_values = round_values[:len(values)]
            else:
                x_values = np.arange(1, len(values) + 1)
            ax.plot(x_values, values, label=key, linewidth=1.8)
    if not ax.lines:
        ax.text(0.5, 0.5, "No training metrics to plot", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Federated Global Training History")
    ax.set_xlabel("Round" if round_values is not None else "Epoch")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def compute_classification_metrics(y_true, y_pred, class_names):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Classification metrics require non-empty prediction and label arrays.")

    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    label_set = list(range(len(class_names)))

    if len(label_set) == 2:
        cm = confusion_matrix(y_true, y_pred, labels=label_set)
        if cm.size == 0:
            raise ValueError("Classification metrics require a valid confusion matrix for the current labels.")
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sensitivity = recall
        balanced_accuracy = (sensitivity + specificity) / 2.0
        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
        }

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_set,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    sensitivities = []
    specificities = []
    for class_index in label_set:
        tp = cm[class_index, class_index]
        fp = cm[:, class_index].sum() - tp
        fn = cm[class_index, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "sensitivity": float(np.mean(sensitivities)) if sensitivities else 0.0,
        "specificity": float(np.mean(specificities)) if specificities else 0.0,
        "balanced_accuracy": float((np.mean(sensitivities) + np.mean(specificities)) / 2.0) if sensitivities and specificities else 0.0,
    }


def plot_confusion_matrix(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    display_labels = []
    for name in class_names:
        normalized = str(name).strip().lower()
        if normalized == "healthy":
            display_labels.append("Healthy")
        elif normalized == "parkinson":
            display_labels.append("Parkinson")
        else:
            display_labels.append(str(name).title())

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found in 'y_true' and 'y_pred'.*",
            category=UserWarning,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    return fig
