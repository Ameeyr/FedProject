import warnings

import numpy as np

from dashboard_utils import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_training_history_by_client,
)


def test_plot_training_history_returns_figure():
    history = {"loss": [1.0, 0.5], "accuracy": [0.2, 0.8]}

    fig = plot_training_history(history)

    assert fig is not None


def test_plot_training_history_uses_round_axis_for_global_history():
    history = {
        "round": [1, 2, 3],
        "global_loss": [0.8, 0.6, 0.5],
        "global_accuracy": [0.70, 0.80, 0.90],
    }

    fig = plot_training_history(history)

    assert fig is not None
    assert fig.axes[0].get_xlabel() == "Round"
    assert fig.axes[0].get_title() == "Federated Global Training History"
    assert list(np.asarray(fig.axes[0].lines[0].get_xdata()).astype(int)) == [1, 2, 3]


def test_plot_training_history_handles_multiple_client_histories():
    histories = [
        {"loss": [1.0, 0.5], "accuracy": [0.2, 0.8]},
        {"loss": [0.9, 0.4], "accuracy": [0.3, 0.9]},
    ]

    fig = plot_training_history(histories)

    assert fig is not None


def test_plot_training_history_by_client_returns_figure():
    histories = [
        {"loss": [1.0, 0.5], "accuracy": [0.2, 0.8]},
        {"loss": [0.9, 0.4], "accuracy": [0.3, 0.9]},
    ]

    fig = plot_training_history_by_client(histories, ["Hospital 1", "Hospital 2"])

    assert fig is not None


def test_plot_training_history_by_client_skips_missing_metrics_and_keeps_epoch_axis():
    histories = [
        {"loss": [1.0, 0.6, 0.4, 0.3, 0.2], "accuracy": [0.5, 0.7, 0.8, 0.9, 0.95]},
        {"loss": [0.9, 0.8, 0.7], "val_loss": [1.1, 0.9, 0.8]},
    ]

    fig = plot_training_history_by_client(histories, ["Hospital 1", "Hospital 2"])

    assert fig is not None
    assert len(fig.axes) == 2
    assert len(fig.axes[0].lines) >= 2
    assert len(fig.axes[1].lines) >= 2
    assert list(fig.axes[0].get_xticks().astype(int))[:1] == [1]


def test_plot_confusion_matrix_returns_figure():
    fig = plot_confusion_matrix(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        ["healthy", "parkinson"],
    )

    assert fig is not None


def test_compute_classification_metrics_returns_clinically_meaningful_values():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    metrics = compute_classification_metrics(y_true, y_pred, ["healthy", "parkinson"])

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert "sensitivity" in metrics
    assert "specificity" in metrics


def test_plot_confusion_matrix_suppresses_single_label_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_confusion_matrix(np.array([0]), np.array([0]), ["healthy", "parkinson"])

    assert not any("single label" in str(w.message).lower() for w in caught)
