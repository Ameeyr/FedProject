def test_hospital_inputs_require_hospital_paths_or_uploaded_images():
    from app import validate_hospital_inputs

    assert validate_hospital_inputs([], []) is False
    assert validate_hospital_inputs(["/tmp/hospital_1"], []) is True
    assert validate_hospital_inputs([], ["image.png"]) is True
