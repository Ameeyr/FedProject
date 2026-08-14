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
