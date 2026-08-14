from pathlib import Path

import numpy as np

from main import load_preprocessed_data
from models.efficientnetb0 import SUPPORTED_MODELS, get_model_head_config


def test_load_preprocessed_data_returns_subset(tmp_path):
    images_path = tmp_path / "preprocessed_images.npy"
    labels_path = tmp_path / "preprocessed_labels.npy"
    class_names_path = tmp_path / "class_names.npy"

    images_data = np.arange(24, dtype=np.float32).reshape(6, 2, 2, 1)
    labels_data = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    class_names_data = np.array(["healthy", "parkinson"], dtype=object)

    np.save(images_path, images_data)
    np.save(labels_path, labels_data)
    np.save(class_names_path, class_names_data)

    images, labels, class_names = load_preprocessed_data(
        images_path,
        labels_path,
        class_names_path,
        max_samples=3,
        seed=7,
    )

    assert images.shape[0] == 3
    assert labels.shape[0] == 3
    assert list(class_names) == ["healthy", "parkinson"]


def test_load_preprocessed_data_caps_large_dataset_by_default(tmp_path):
    images_path = tmp_path / "preprocessed_images.npy"
    labels_path = tmp_path / "preprocessed_labels.npy"
    class_names_path = tmp_path / "class_names.npy"

    images_data = np.arange(12000, dtype=np.float32).reshape(3000, 2, 2, 1)
    labels_data = np.tile(np.array([0, 1], dtype=np.int32), 1500)
    class_names_data = np.array(["healthy", "parkinson"], dtype=object)

    np.save(images_path, images_data)
    np.save(labels_path, labels_data)
    np.save(class_names_path, class_names_data)

    images, labels, _ = load_preprocessed_data(
        images_path,
        labels_path,
        class_names_path,
    )

    assert images.shape[0] == 2048
    assert labels.shape[0] == 2048


def test_supported_models_include_transfer_backbones():
    assert "efficientnetb0" in SUPPORTED_MODELS
    assert "resnet50" in SUPPORTED_MODELS
    assert "mobilenetv2" in SUPPORTED_MODELS


def test_binary_classification_uses_single_sigmoid_output():
    config = get_model_head_config(2)

    assert config["units"] == 1
    assert config["activation"] == "sigmoid"
    assert config["loss"] == "binary"


def test_single_class_datasets_use_binary_sigmoid_output():
    config = get_model_head_config(1)

    assert config["units"] == 1
    assert config["activation"] == "sigmoid"
    assert config["loss"] == "binary"
