import pytest


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, class_id, conf=1.0):
        self.cls = FakeScalar(class_id)
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


def make_detector(module, results, min_confidence=0.25):
    detector = module.CarDetector.__new__(module.CarDetector)
    detector.min_confidence = min_confidence
    detector.car_detection_model = FakeModel(results)
    return detector


def test_detect_car_returns_true_for_car(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(0)],
        names={0: "Car"},
    )
    detector = make_detector(image_processor_module, [result])
    image = object()

    assert detector.detect_car(image) is True
    assert detector.car_detection_model.calls == [(image, False, 0.25)]


def test_detect_car_returns_true_for_truck(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(3)],
        names={3: "TRUCK"},
    )
    detector = make_detector(image_processor_module, [result])

    assert detector.detect_car(object()) is True


def test_detect_car_returns_false_when_no_boxes(image_processor_module):
    result = FakeResult(boxes=[], names={})
    detector = make_detector(image_processor_module, [result])

    assert detector.detect_car(object()) is False


def test_detect_car_returns_false_for_other_object(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(1), FakeBox(2)],
        names={1: "person", 2: "bicycle"},
    )
    detector = make_detector(image_processor_module, [result])

    assert detector.detect_car(object()) is False


def test_detect_car_ignores_boxes_below_min_confidence(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(0, conf=0.4)],
        names={0: "car"},
    )
    detector = make_detector(image_processor_module, [result], min_confidence=0.5)

    assert detector.detect_car(object()) is False
    assert detector.car_detection_model.calls[0][2] == 0.5


def test_detect_car_accepts_box_at_min_confidence(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(0, conf=0.3), FakeBox(1, conf=0.5)],
        names={0: "person", 1: "truck"},
    )
    detector = make_detector(image_processor_module, [result], min_confidence=0.5)

    assert detector.detect_car(object()) is True


def test_car_detector_default_min_confidence(monkeypatch, image_processor_module):
    monkeypatch.setattr(image_processor_module, "YOLO", lambda path: FakeModel([]))

    detector = image_processor_module.CarDetector()

    assert detector.min_confidence == 0.25


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), "0.5", None, True])
def test_car_detector_rejects_invalid_min_confidence(
    monkeypatch, image_processor_module, value,
):
    loaded = []
    monkeypatch.setattr(image_processor_module, "YOLO", lambda path: loaded.append(path))

    with pytest.raises((TypeError, ValueError)):
        image_processor_module.CarDetector(min_confidence=value)
    # некорректный порог отклоняется до загрузки модели
    assert loaded == []
