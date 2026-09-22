import math
from dataclasses import dataclass
from numbers import Real

# Значения по умолчанию подобраны так, чтобы поведение pipeline не менялось:
# 0.25 — внутренний порог YOLO по умолчанию, 0.0 — сцена возвращается целиком.
DEFAULT_CAR_CONFIDENCE = 0.25
DEFAULT_OBJECT_CONFIDENCE = 0.25
DEFAULT_SCENE_CONFIDENCE = 0.0


def validate_confidence(value, name: str = "confidence") -> float:
    """
    Проверяет порог уверенности и возвращает его как float.

    :raises TypeError: если значение не является числом (bool тоже не допускается)
    :raises ValueError: если значение NaN/inf или лежит вне отрезка [0, 1]
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} должен быть числом от 0 до 1, получено {type(value).__name__}: {value!r}")

    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} должен лежать в диапазоне [0, 1], получено {value!r}")

    return value


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Минимальные пороги уверенности для компонентов pipeline."""
    car: float = DEFAULT_CAR_CONFIDENCE
    objects: float = DEFAULT_OBJECT_CONFIDENCE
    scene: float = DEFAULT_SCENE_CONFIDENCE

    def __post_init__(self):
        for field_name in ("car", "objects", "scene"):
            value = validate_confidence(getattr(self, field_name), f"ConfidenceThresholds.{field_name}")
            object.__setattr__(self, field_name, value)
