import pytest


class FakeBox:
    def __init__(self, class_id, conf):
        self.cls = class_id
        self.conf = conf


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, image, verbose=False, conf=None):
        self.calls.append((image, verbose, conf))
        return self.results


def make_object_extractor(module, results, min_confidence):
    extractor = module.ObjectExtractor.__new__(module.ObjectExtractor)
    extractor.min_confidence = min_confidence
    extractor.object_detection_model = FakeModel(results)
    return extractor


def make_scene_extractor(module, min_confidence):
    extractor = module.SceneExtractor.__new__(module.SceneExtractor)
    extractor.min_confidence = min_confidence
    return extractor


def test_extract_features_filters_by_min_confidence(feature_extractor_module):
    result = FakeResult(
        boxes=[FakeBox(0, 0.9), FakeBox(1, 0.49), FakeBox(2, 0.5)],
        names={0: "Tree", 1: "Person", 2: "Building"},
    )
    extractor = make_object_extractor(feature_extractor_module, [result], 0.5)
    image = object()

    assert extractor.extract_features(image) == {"Tree": 0.9, "Building": 0.5}
    assert extractor.object_detection_model.calls == [(image, False, 0.5)]


def test_extract_features_with_zero_threshold_keeps_everything(feature_extractor_module):
    result = FakeResult(
        boxes=[FakeBox(0, 0.01), FakeBox(1, 0.3)],
        names={0: "Tree", 1: "Person"},
    )
    extractor = make_object_extractor(feature_extractor_module, [result], 0.0)

    assert extractor.extract_features(object()) == {"Tree": 0.01, "Person": 0.3}


def test_scene_filter_scores(feature_extractor_module):
    extractor = make_scene_extractor(feature_extractor_module, 0.5)

    scores = {"highway": 0.2, "residential": 0.5, "parking lot": 0.75}

    assert extractor.filter_scores(scores) == {"residential": 0.5, "parking lot": 0.75}


def test_scene_filter_scores_default_keeps_all(feature_extractor_module):
    extractor = make_scene_extractor(
        feature_extractor_module,
        feature_extractor_module.DEFAULT_SCENE_CONFIDENCE,
    )
    scores = {"other": 0.0, "highway": 0.1}

    assert extractor.filter_scores(scores) == scores


def test_extractors_default_min_confidence(monkeypatch, feature_extractor_module):
    monkeypatch.setattr(feature_extractor_module, "YOLO", lambda path: FakeModel([]))
    monkeypatch.setattr(
        feature_extractor_module.SceneExtractor,
        "_SceneExtractor__init_scene_classification_model",
        staticmethod(lambda: object()),
    )

    assert feature_extractor_module.ObjectExtractor().min_confidence == 0.25
    assert feature_extractor_module.SceneExtractor().min_confidence == 0.0


@pytest.mark.parametrize("value", [-0.5, 1.2, float("nan"), "0.3", None])
def test_extractors_reject_invalid_min_confidence(monkeypatch, feature_extractor_module, value):
    loaded = []
    monkeypatch.setattr(feature_extractor_module, "YOLO", lambda path: loaded.append(path))
    monkeypatch.setattr(
        feature_extractor_module.SceneExtractor,
        "_SceneExtractor__init_scene_classification_model",
        staticmethod(lambda: loaded.append("scene")),
    )

    with pytest.raises((TypeError, ValueError)):
        feature_extractor_module.ObjectExtractor(min_confidence=value)
    with pytest.raises((TypeError, ValueError)):
        feature_extractor_module.SceneExtractor(min_confidence=value)
    assert loaded == []
