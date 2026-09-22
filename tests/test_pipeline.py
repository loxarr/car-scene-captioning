import sys
import types

import pytest

from conftest import load_package_module, load_source_module


def load_main_with_fakes(monkeypatch, *, loaded_image, has_car=True, scene=None):
    if scene is None:
        # main.py drops the "tunnel" class from the scene result.
        scene = {"city street": 0.81, "tunnel": 1.0}
    thresholds = {}
    counters = {
        "ImageRotator": 0,
        "CarDetector": 0,
        "ObjectExtractor": 0,
        "SceneExtractor": 0,
        "PhotoDescriber": 0,
        "PhotoDescriberWithQuestion": 0,
        "ImageLoader": 0,
        "rotate": 0,
        "detect_car": 0,
        "extract_features": 0,
        "predict_scene": 0,
        "simple_photo_describe": 0,
        "photo_describe": 0,
    }

    class ImageLoader:
        def __init__(self, path, source):
            counters["ImageLoader"] += 1
            self.path = path
            self.source = source

        def load_image(self):
            return loaded_image(self.path) if callable(loaded_image) else loaded_image

    class ImageRotator:
        def __init__(self):
            counters["ImageRotator"] += 1

        def rotate_image(self, image):
            counters["rotate"] += 1
            return image

    class CarDetector:
        def __init__(self, min_confidence=None):
            counters["CarDetector"] += 1
            thresholds["car"] = min_confidence

        def detect_car(self, image):
            counters["detect_car"] += 1
            return has_car

    class ObjectExtractor:
        def __init__(self, min_confidence=None):
            counters["ObjectExtractor"] += 1
            thresholds["objects"] = min_confidence

        def extract_features(self, image):
            counters["extract_features"] += 1
            return {"Tree": 2, "Building": 1}

    class SceneExtractor:
        def __init__(self, min_confidence=None):
            counters["SceneExtractor"] += 1
            thresholds["scene"] = min_confidence

        def predict_scene(self, image):
            counters["predict_scene"] += 1
            return dict(scene)

    class PhotoDescriber:
        def __init__(self):
            counters["PhotoDescriber"] += 1

        def simple_photo_describe(self, image):
            counters["simple_photo_describe"] += 1
            return {
                "base": "base",
                "detailed": "detailed",
                "alternative": "alternative",
            }

    class PhotoDescriberWithQuestion:
        def __init__(self):
            counters["PhotoDescriberWithQuestion"] += 1

        def photo_describe(self, image):
            counters["photo_describe"] += 1
            return "vqa"

    fake_autocaption = types.ModuleType("autocaption")
    fake_autocaption.ImageLoader = ImageLoader
    fake_autocaption.ImageRotator = ImageRotator
    fake_autocaption.CarDetector = CarDetector
    fake_autocaption.ObjectExtractor = ObjectExtractor
    fake_autocaption.SceneExtractor = SceneExtractor
    fake_autocaption.PhotoDescriber = PhotoDescriber
    fake_autocaption.PhotoDescriberWithQuestion = PhotoDescriberWithQuestion
    # configuration object has no heavy dependencies, so the real one is used
    fake_autocaption.ConfidenceThresholds = load_package_module("confidence").ConfidenceThresholds

    monkeypatch.setitem(sys.modules, "autocaption", fake_autocaption)

    main = load_source_module("_main_under_test", "main.py")
    counters["thresholds"] = thresholds
    return main, counters


def test_pipeline_success(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        has_car=True,
    )

    result = main.run_pipeline(["car.jpg"], source=True)

    assert result == [[
        {"Tree": 2, "Building": 1},
        {"city street": 0.81},
        "Базовое описание: base",
        "Подробное описание: detailed",
        "Альтернативное описание: alternative",
        "Описание по вопросу (VQA): vqa",
    ]]

    assert counters["ImageLoader"] == 1
    assert counters["rotate"] == 1
    assert counters["detect_car"] == 1
    assert counters["extract_features"] == 1
    assert counters["predict_scene"] == 1
    assert counters["simple_photo_describe"] == 1
    assert counters["photo_describe"] == 1


def test_pipeline_skips_remaining_steps_when_image_cannot_be_loaded(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=None,
        has_car=True,
    )

    result = main.run_pipeline(["missing.jpg"], source=True)

    assert result == [[
        "Не удалось загрузить изображение: missing.jpg."
    ]]
    assert counters["rotate"] == 0
    assert counters["detect_car"] == 0
    assert counters["extract_features"] == 0
    assert counters["predict_scene"] == 0
    assert counters["simple_photo_describe"] == 0
    assert counters["photo_describe"] == 0


def test_pipeline_stops_when_car_is_not_detected(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        has_car=False,
    )

    result = main.run_pipeline(["no-car.jpg"], source=True)

    assert result == [[
        "Не удалось определить наличие автомобиля на изображении: no-car.jpg."
    ]]
    assert counters["rotate"] == 1
    assert counters["detect_car"] == 1
    assert counters["extract_features"] == 0
    assert counters["predict_scene"] == 0
    assert counters["simple_photo_describe"] == 0
    assert counters["photo_describe"] == 0


def test_heavy_components_are_initialized_once_for_multiple_images(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        has_car=True,
    )

    main.run_pipeline(["one.jpg", "two.jpg"], source=True)

    assert counters["ImageRotator"] == 1
    assert counters["CarDetector"] == 1
    assert counters["ObjectExtractor"] == 1
    assert counters["SceneExtractor"] == 1
    assert counters["PhotoDescriber"] == 1
    assert counters["PhotoDescriberWithQuestion"] == 1

    assert counters["ImageLoader"] == 2
    assert counters["rotate"] == 2
    assert counters["detect_car"] == 2


def test_pipeline_uses_default_thresholds(monkeypatch):
    main, counters = load_main_with_fakes(monkeypatch, loaded_image=object())

    main.run_pipeline(["car.jpg"], source=True)

    assert counters["thresholds"] == {"car": 0.25, "objects": 0.25, "scene": 0.0}


def test_pipeline_passes_custom_thresholds_to_components(monkeypatch):
    main, counters = load_main_with_fakes(monkeypatch, loaded_image=object())

    thresholds = main.ConfidenceThresholds(car=0.7, objects=0.4, scene=0.5)
    main.run_pipeline(["one.jpg", "two.jpg"], source=True, confidence=thresholds)

    assert counters["thresholds"] == {"car": 0.7, "objects": 0.4, "scene": 0.5}
    assert counters["CarDetector"] == 1
    assert counters["ObjectExtractor"] == 1
    assert counters["SceneExtractor"] == 1


@pytest.mark.parametrize("value", [0.5, {"car": 0.5}, "0.5"])
def test_pipeline_rejects_invalid_confidence_before_loading_models(monkeypatch, value):
    main, counters = load_main_with_fakes(monkeypatch, loaded_image=object())

    with pytest.raises(TypeError, match="ConfidenceThresholds"):
        main.run_pipeline(["car.jpg"], source=True, confidence=value)

    assert counters["ImageRotator"] == 0
    assert counters["CarDetector"] == 0
    assert counters["ImageLoader"] == 0


def test_pipeline_keeps_scene_classes_when_tunnel_was_filtered_out(monkeypatch):
    # with a scene threshold "tunnel" may be absent; other classes must survive
    main, _ = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        scene={"residential": 0.9, "parking lot": 0.75},
    )

    result = main.run_pipeline(["car.jpg"], source=True)

    assert result[0][1] == {"residential": 0.9, "parking lot": 0.75}
