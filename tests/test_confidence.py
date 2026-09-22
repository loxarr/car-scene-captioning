import dataclasses
import math

import pytest


def test_validate_confidence_accepts_bounds_and_ints(confidence_module):
    validate = confidence_module.validate_confidence

    assert validate(0) == 0.0
    assert validate(1) == 1.0
    assert validate(0.42) == 0.42
    assert isinstance(validate(1), float)


@pytest.mark.parametrize("value", [-0.01, 1.01, 100, float("nan"), float("inf"), -math.inf])
def test_validate_confidence_rejects_out_of_range(confidence_module, value):
    with pytest.raises(ValueError, match="car_conf"):
        confidence_module.validate_confidence(value, "car_conf")


@pytest.mark.parametrize("value", ["0.5", None, True, False, [0.5]])
def test_validate_confidence_rejects_non_numbers(confidence_module, value):
    with pytest.raises(TypeError):
        confidence_module.validate_confidence(value)


def test_thresholds_defaults(confidence_module):
    thresholds = confidence_module.ConfidenceThresholds()

    assert thresholds.car == confidence_module.DEFAULT_CAR_CONFIDENCE == 0.25
    assert thresholds.objects == confidence_module.DEFAULT_OBJECT_CONFIDENCE == 0.25
    assert thresholds.scene == confidence_module.DEFAULT_SCENE_CONFIDENCE == 0.0


def test_thresholds_custom_values(confidence_module):
    thresholds = confidence_module.ConfidenceThresholds(car=0.6, objects=1, scene=0.3)

    assert (thresholds.car, thresholds.objects, thresholds.scene) == (0.6, 1.0, 0.3)
    assert isinstance(thresholds.objects, float)


@pytest.mark.parametrize("field", ["car", "objects", "scene"])
@pytest.mark.parametrize("value, error", [(-1, ValueError), (2, ValueError), ("high", TypeError)])
def test_thresholds_reject_invalid_values(confidence_module, field, value, error):
    with pytest.raises(error, match=f"ConfidenceThresholds.{field}"):
        confidence_module.ConfidenceThresholds(**{field: value})


def test_thresholds_are_immutable(confidence_module):
    thresholds = confidence_module.ConfidenceThresholds()

    with pytest.raises(dataclasses.FrozenInstanceError):
        thresholds.car = 2
