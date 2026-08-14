import numpy as np

from federated.server import partition_dataset


def test_partition_dataset_balances_class_labels():
    images = np.arange(12).reshape(12, 1, 1, 1)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)

    partitions = partition_dataset(images, labels, num_clients=3, seed=7)

    assert len(partitions) == 3
    for client_images, client_labels in partitions:
        assert client_images.shape[0] == client_labels.shape[0]
        assert client_labels.size > 0

    combined_labels = np.concatenate([client_labels for _, client_labels in partitions])
    assert np.array_equal(np.sort(combined_labels), np.sort(labels))
