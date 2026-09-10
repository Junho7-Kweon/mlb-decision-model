from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .features import FEATURE_NAMES, FEATURE_VERSION


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class CalibratedLogisticModel:
    """Small dependency-free logistic model with Platt calibration."""

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES):
        self.feature_names = tuple(feature_names)
        self.means = [0.0] * len(self.feature_names)
        self.scales = [1.0] * len(self.feature_names)
        self.weights = [0.0] * len(self.feature_names)
        self.intercept = 0.0
        self.calibration_a = 1.0
        self.calibration_b = 0.0

    def _vector(self, row: dict[str, float]) -> list[float]:
        return [float(row.get(name, 0.0)) for name in self.feature_names]

    def _standardize(self, vector: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(vector, self.means, self.scales)]

    def fit(
        self,
        rows: Sequence[dict[str, float]],
        labels: Sequence[int],
        epochs: int = 1800,
        learning_rate: float = 0.035,
        l2: float = 0.02,
    ) -> "CalibratedLogisticModel":
        if len(rows) != len(labels) or len(rows) < 20:
            raise ValueError("At least 20 aligned historical games are required")
        matrix = [self._vector(row) for row in rows]
        n = len(matrix)
        split = max(16, int(n * 0.8))
        for j in range(len(self.feature_names)):
            values = [row[j] for row in matrix[:split]]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.means[j] = mean
            self.scales[j] = max(1e-6, math.sqrt(variance))
        x = [self._standardize(row) for row in matrix]
        split = max(16, int(n * 0.8))
        train_x, train_y = x[:split], labels[:split]
        valid_x, valid_y = x[split:], labels[split:]
        if n >= 1000:
            try:
                import numpy as np
            except ImportError:
                pass
            else:
                design = np.asarray(train_x, dtype=float)
                targets = np.asarray(train_y, dtype=float)
                weights = np.asarray(self.weights, dtype=float)
                for _ in range(epochs):
                    logits = np.clip(self.intercept + design @ weights, -700, 700)
                    errors = 1 / (1 + np.exp(-logits)) - targets
                    self.intercept -= learning_rate * float(errors.mean())
                    weights -= learning_rate * (design.T @ errors / len(targets) + l2 * weights)
                self.weights = weights.tolist()
                if len(valid_x) >= 4:
                    logits = [self.intercept + sum(w*v for w,v in zip(self.weights,row)) for row in valid_x]
                    self._fit_calibrator(logits, valid_y)
                return self
        for _ in range(epochs):
            grad_w = [0.0] * len(self.weights)
            grad_b = 0.0
            for vector, target in zip(train_x, train_y):
                pred = _sigmoid(self.intercept + sum(w * v for w, v in zip(self.weights, vector)))
                error = pred - int(target)
                grad_b += error
                for j, value in enumerate(vector):
                    grad_w[j] += error * value
            size = max(1, len(train_x))
            self.intercept -= learning_rate * grad_b / size
            for j in range(len(self.weights)):
                penalty = l2 * self.weights[j]
                self.weights[j] -= learning_rate * (grad_w[j] / size + penalty)
        if len(valid_x) >= 4:
            logits = [self.intercept + sum(w * v for w, v in zip(self.weights, row)) for row in valid_x]
            self._fit_calibrator(logits, valid_y)
        return self

    def _fit_calibrator(self, logits: Sequence[float], labels: Sequence[int]) -> None:
        a, b = 1.0, 0.0
        for _ in range(700):
            grad_a = 0.0
            grad_b = 0.0
            for logit, target in zip(logits, labels):
                error = _sigmoid(a * logit + b) - int(target)
                grad_a += error * logit
                grad_b += error
            n = max(1, len(logits))
            a -= 0.02 * grad_a / n
            b -= 0.02 * grad_b / n
        self.calibration_a = max(0.05, min(5.0, a))
        self.calibration_b = max(-3.0, min(3.0, b))

    def predict_home_win(self, features: dict[str, float]) -> float:
        vector = self._standardize(self._vector(features))
        logit = self.intercept + sum(w * v for w, v in zip(self.weights, vector))
        return _sigmoid(self.calibration_a * logit + self.calibration_b)

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_version": FEATURE_VERSION,
            "feature_names": self.feature_names,
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "intercept": self.intercept,
            "calibration_a": self.calibration_a,
            "calibration_b": self.calibration_b,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedLogisticModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact_names = tuple(payload.get("feature_names", ()))
        artifact_version = payload.get("feature_version")
        if artifact_version != FEATURE_VERSION or artifact_names != FEATURE_NAMES:
            raise ValueError(
                "Model artifact uses an incompatible feature schema; "
                f"expected {FEATURE_VERSION} {FEATURE_NAMES}, got "
                f"{artifact_version!r} {artifact_names}. Retrain the model."
            )
        model = cls(artifact_names)
        for key in ("means", "scales", "weights", "intercept", "calibration_a", "calibration_b"):
            setattr(model, key, payload[key])
        return model


def brier_score(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    pairs = list(zip(probabilities, labels))
    return sum((p - y) ** 2 for p, y in pairs) / max(1, len(pairs))
