import numpy as np

import app


def test_streamlit_compat_ignores_unsupported_container_width(monkeypatch):
    seen = []

    class DummyStreamlit:
        @staticmethod
        def image(*args, **kwargs):
            seen.append(kwargs)
            if "use_container_width" in kwargs:
                raise TypeError("ImageMixin.image() got an unexpected keyword argument 'use_container_width'")
            return "image-ok"

        @staticmethod
        def dataframe(*args, **kwargs):
            seen.append(kwargs)
            if "use_container_width" in kwargs:
                raise TypeError("DataFrameMixin.dataframe() got an unexpected keyword argument 'use_container_width'")
            return "dataframe-ok"

    monkeypatch.setattr(app, "st", DummyStreamlit())

    assert app.st_image_compat(np.zeros((2, 2, 3), dtype=np.uint8), caption="demo") == "image-ok"
    assert app.st_dataframe_compat([{"a": 1}], width=300) == "dataframe-ok"
    assert any("use_container_width" in kwargs for kwargs in seen if isinstance(kwargs, dict)) is False
