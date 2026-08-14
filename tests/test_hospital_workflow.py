import numpy as np

from app import parse_hospital_dataset_paths, remap_labels_to_shared_classes


def test_parse_hospital_dataset_paths_handles_multiple_lines():
    raw_value = " /tmp/hospital_a \n /tmp/hospital_b \n\n"
    parsed = parse_hospital_dataset_paths(raw_value)
    assert parsed == ["/tmp/hospital_a", "/tmp/hospital_b"]


def test_remap_labels_to_shared_classes_maps_local_indices():
    labels = np.array([0, 1, 1], dtype=np.int32)
    class_names = ["NonPDpatients", "PDpateints"]
    shared_class_names = ["PDpateints", "NonPDpatients"]

    remapped = remap_labels_to_shared_classes(labels, class_names, shared_class_names)

    assert np.array_equal(remapped, np.array([1, 0, 0], dtype=np.int32))
