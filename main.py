import argparse
from pathlib import Path

import numpy as np

from models.efficientnetb0 import FederatedClient, SUPPORTED_MODELS


def load_preprocessed_data(images_path, labels_path, class_names_path, max_samples=None, seed=42):
    if not images_path.exists() or not labels_path.exists() or not class_names_path.exists():
        raise FileNotFoundError("Preprocessed arrays were not found. Run the preprocessing step first.")

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    class_names = np.load(class_names_path, allow_pickle=True)

    if len(images) == 0:
        raise ValueError("No preprocessed images were found.")

    if max_samples is None:
        max_samples = 2048

    max_samples = max(2, int(max_samples))
    if len(images) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(images))[:max_samples]
        images = images[indices]
        labels = labels[indices]

    num_classes = len(class_names)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    return images, labels, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transfer-learning model on the preprocessed image arrays")
    parser.add_argument(
        "--model",
        default="efficientnetb0",
        choices=sorted(SUPPORTED_MODELS),
        help="Model backbone to use",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for quick runs (defaults to 2048 when omitted)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    return parser.parse_args()


def main(model_name="efficientnetb0", max_samples=None, epochs=1, batch_size=2):
    project_dir = Path(__file__).resolve().parent
    result_dir = project_dir / "result"
    images_path = result_dir / "preprocessed_images.npy"
    labels_path = result_dir / "preprocessed_labels.npy"
    class_names_path = result_dir / "class_names.npy"

    images, labels, class_names = load_preprocessed_data(
        images_path,
        labels_path,
        class_names_path,
        max_samples=max_samples,
    )

    num_classes = len(class_names)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(images))
    split_index = max(1, int(0.8 * len(images)))

    train_idx = indices[:split_index]
    val_idx = indices[split_index:]

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    val_images = images[val_idx]
    val_labels = labels[val_idx]

    client = FederatedClient(client_id=1, server_address="local", model_name=model_name, num_classes=num_classes)
    history = client.train_model(
        (train_images, train_labels),
        (val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
    )
    accuracy = client.evaluate_model((val_images, val_labels))

    print(f"Loaded {len(images)} samples across {num_classes} classes.")
    print(f"Model: {model_name}")
    print(f"Class names: {list(class_names)}")
    print(f"Final validation accuracy: {accuracy:.4f}")
    return history, accuracy


if __name__ == "__main__":
    args = parse_args()
    main(model_name=args.model, max_samples=args.max_samples, epochs=args.epochs, batch_size=args.batch_size)
