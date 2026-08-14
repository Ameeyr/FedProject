from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.preprocess_images import ImagePreprocessor


def test_chunk_size_limits_preprocessed_images(tmp_path):
    class_a_dir = tmp_path / "class_a"
    class_b_dir = tmp_path / "class_b"
    class_a_dir.mkdir()
    class_b_dir.mkdir()

    for cls_dir in [class_a_dir, class_b_dir]:
        for idx in range(3):
            image_path = cls_dir / f"image_{idx}.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)

    preprocessor = ImagePreprocessor(
        str(tmp_path),
        target_size=(16, 16),
        batch_size=2,
        output_dir=str(tmp_path / "output"),
        chunk_size=4,
    )

    images, labels, class_names = preprocessor.preprocess_images()

    assert images.shape[0] == 4
    assert labels.shape[0] == 4
    assert class_names == ["class_a", "class_b"]
