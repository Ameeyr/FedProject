from app import resolve_hospital_selection


def test_resolve_hospital_selection_uses_all_provided_paths():
    hospital_datasets = [{"path": "h1"}, {"path": "h2"}, {"path": "h3"}]

    selected, used_count = resolve_hospital_selection(hospital_datasets, ["h1", "h2", "h3"], 2)

    assert used_count == 3
    assert [d["path"] for d in selected] == ["h1", "h2", "h3"]


def test_resolve_hospital_selection_caps_to_available_paths():
    hospital_datasets = [{"path": "h1"}, {"path": "h2"}]

    selected, used_count = resolve_hospital_selection(hospital_datasets, ["h1", "h2"], 5)

    assert used_count == 2
    assert [d["path"] for d in selected] == ["h1", "h2"]
