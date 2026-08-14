import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


class ImagePreprocessor:
    def __init__(self, dataset_path, target_size=(224, 224), batch_size=16, output_dir=None, chunk_size=None):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.batch_size = max(1, int(batch_size))
        self.chunk_size = max(1, int(chunk_size)) if chunk_size is not None else None
        self.output_dir = output_dir or str(Path(__file__).resolve().parents[2] / "result")

    @staticmethod
    def _is_supported_image(file_name):
        return os.path.splitext(file_name)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}

    @staticmethod
    def _get_class_name(dataset_path, file_path):
        relative_path = os.path.relpath(file_path, dataset_path)
        parts = Path(relative_path).parts
        if not parts:
            return None
        return parts[0]

    def preprocess_images(self):
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        os.makedirs(self.output_dir, exist_ok=True)

        image_paths = []
        class_names = []
        class_to_index = {}

        for root, _, files in sorted(os.walk(self.dataset_path), key=lambda item: item[0]):
            if not files:
                continue

            for image_name in sorted(files):
                if not self._is_supported_image(image_name):
                    continue

                image_path = os.path.join(root, image_name)
                class_name = self._get_class_name(self.dataset_path, image_path)
                if class_name is None:
                    continue

                if class_name not in class_to_index:
                    class_to_index[class_name] = len(class_names)
                    class_names.append(class_name)

                image_paths.append((image_path, class_to_index[class_name]))

        if self.chunk_size is not None and len(image_paths) > self.chunk_size:
            image_paths = image_paths[: self.chunk_size]

        image_shape = (len(image_paths), self.target_size[0], self.target_size[1], 3)
        images_path = os.path.join(self.output_dir, "preprocessed_images.npy")
        labels_path = os.path.join(self.output_dir, "preprocessed_labels.npy")
        class_names_path = os.path.join(self.output_dir, "class_names.npy")

        images_memmap = np.lib.format.open_memmap(
            images_path,
            mode="w+",
            dtype="float32",
            shape=image_shape,
        )
        labels_memmap = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype="int32",
            shape=(len(image_paths),),
        )

        write_index = 0
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            batch_images = []
            batch_labels = []

            for image_path, label in batch_paths:
                try:
                    with PILImage.open(image_path) as img:
                        img = img.convert("RGB")
                        img = img.resize(self.target_size)
                        img_array = np.array(img, dtype="float32")
                        img_array = img_array / 255.0
                    batch_images.append(img_array)
                    batch_labels.append(label)
                except Exception as exc:
                    print(f"Error processing image {image_path}: {exc}")

            if not batch_images:
                continue

            batch_images_array = np.stack(batch_images, axis=0).astype("float32")
            batch_labels_array = np.array(batch_labels, dtype="int32")
            count = batch_images_array.shape[0]
            images_memmap[write_index : write_index + count] = batch_images_array
            labels_memmap[write_index : write_index + count] = batch_labels_array
            write_index += count

        np.save(class_names_path, np.array(class_names, dtype=object))
        images_memmap = images_memmap[:write_index]
        labels_memmap = labels_memmap[:write_index]
        return images_memmap, labels_memmap, class_names


if __name__ == "__main__":
    dataset_path = str(Path(__file__).resolve().parents[2] / "dataset")
    output_dir = str(Path(__file__).resolve().parents[2] / "result")
    preprocessor = ImagePreprocessor(dataset_path, output_dir=output_dir, batch_size=16)
    images, labels, class_names = preprocessor.preprocess_images()

    print(f"Preprocessed {len(images)} images with {len(class_names)} classes.")
    print(f"Class names: {class_names}")
    print(f"Saved arrays to {output_dir}")
