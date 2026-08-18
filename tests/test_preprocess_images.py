import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.preprocess_images import ImagePreprocessor


def test_preprocess_images_saves_batchwise(tmp_path):
    class_a_dir = tmp_path / "class_a"
    class_b_dir = tmp_path / "class_b"
    class_a_dir.mkdir()
    class_b_dir.mkdir()

    for cls_dir in [class_a_dir, class_b_dir]:
        for idx in range(2):
            image_path = cls_dir / f"image_{idx}.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)

    preprocessor = ImagePreprocessor(
        str(tmp_path),
        target_size=(16, 16),
        batch_size=1,
        output_dir=str(tmp_path / "output"),
    )

    images, labels, class_names = preprocessor.preprocess_images()

    assert images.shape[0] == 4
    assert labels.shape[0] == 4
    assert images.shape[1:] == (16, 16, 3)
    assert class_names == ["class_a", "class_b"]
    assert np.array_equal(np.unique(labels), np.array([0, 1]))


def test_preprocess_images_uses_top_level_class_folders_for_nested_data(tmp_path):
    healthy_dir = tmp_path / "healthy" / "Subject1" / "1.MRI"
    parkinson_dir = tmp_path / "parkinson" / "Subject2" / "1.MRI"
    healthy_dir.mkdir(parents=True)
    parkinson_dir.mkdir(parents=True)

    for cls_dir in [healthy_dir, parkinson_dir]:
        for idx in range(2):
            image_path = cls_dir / f"image_{idx}.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)

    preprocessor = ImagePreprocessor(
        str(tmp_path),
        target_size=(16, 16),
        batch_size=1,
        output_dir=str(tmp_path / "output"),
    )

    images, labels, class_names = preprocessor.preprocess_images()

    assert images.shape[0] == 4
    assert labels.shape[0] == 4
    assert class_names == ["healthy", "parkinson"]
    assert np.array_equal(np.unique(labels), np.array([0, 1]))


def test_split_train_val_keeps_both_classes_in_validation():
    images = np.zeros((20, 8, 8, 3), dtype=np.float32)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int32)

    from app import split_train_val

    train_data, val_data = split_train_val(images, labels, val_fraction=0.2, seed=7)
    train_images, train_labels = train_data
    val_images, val_labels = val_data

    assert set(np.unique(train_labels)) == {0, 1}
    assert set(np.unique(val_labels)) == {0, 1}
    assert len(val_labels) > 0
    assert len(train_images) + len(val_images) == len(images)
