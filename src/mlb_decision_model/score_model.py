from __future__ import annotations

"""Joint score-distribution model: P(away_runs=x, home_runs=y | pregame features).

이게 저희가 처음부터 목표로 잡았던 "진짜" 모델입니다 - model.py의
CalibratedLogisticModel(홈 승률 하나만 냄)과 달리, 여기서는 홈·원정 득점을
각각 예측해서 점수 결합분포를 만들고, 그 분포에서 승패·언더오버·핸디캡
확률을 전부 파생시킵니다.

*** 설계상 중요한 제약 ***
features.py의 9개 피처는 전부 "홈-원정 차이"(signed edge) 형태입니다.
득점분포 모델은 홈 득점과 원정 득점을 "따로" 예측해야 하므로, 같은 9개
피처를 입력으로 두 개의 독립된 포아송 회귀(홈용, 원정용)에 넣습니다 -
회귀 자체가 "이 피처가 홈 득점에는 이렇게, 원정 득점에는 저렇게 영향을
준다"를 각자 다른 가중치로 학습합니다.

*** 현재 분포 ***
- 평균은 두 개의 포아송 회귀로 예측하되, 실제 득점의 과산포를 반영하도록
  최종 PMF는 음이항분포(Negative Binomial)를 사용한다.
- 득점 결합분포는 "홈·원정 독립" 가정(baseline A). 상관 반영(bivariate)은
  이후 단계.
- 마켓 변환은 승패·언더오버·핸디캡 3개만 구현. 승1패·홀짝은 정산 규칙이
  이 사이트 기준으로 아직 명확히 확인 안 돼서(특히 "1"이 정확히 뭘
  의미하는지) 일부러 뺐다 - 잘못된 규칙으로 확률을 내는 것보다 안 내는
  게 안전하다.
"""

import json
import math
from pathlib import Path
from typing import Sequence

from .features import FEATURE_NAMES, FEATURE_VERSION


MAX_RUNS = 15  # Minimum grid extent; expand until omitted probability < 1e-10.


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _nb_pmf(k: int, mu: float, dispersion: float) -> float:
    """Negative-binomial PMF with Var(Y) = mu + mu^2 / dispersion."""
    if k < 0 or not all(math.isfinite(v) for v in (mu, dispersion)):
        return 0.0
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    if dispersion <= 0:
        raise ValueError("dispersion must be positive")
    # The NB converges to Poisson as r grows.  Evaluating lgamma(k+r)-lgamma(r)
    # directly at very large r loses enough floating-point precision to keep a
    # tail-expansion loop alive forever, so use the exact limiting calculation.
    if dispersion >= 1e5:
        return _poisson_pmf(k, mu)
    r = dispersion
    return math.exp(
        math.lgamma(k + r)
        - math.lgamma(r)
        - math.lgamma(k + 1)
        + r * math.log(r / (r + mu))
        + k * math.log(mu / (r + mu))
    )


def estimate_dispersion(
    actuals: Sequence[int], predicted_means: Sequence[float]
) -> float:
    """Estimate pooled NB dispersion from training residuals (method of moments)."""
    if len(actuals) != len(predicted_means) or not actuals:
        raise ValueError("Aligned score labels and predicted means are required")
    numerator = sum(float(mu) ** 2 for mu in predicted_means)
    denominator = sum(
        (float(y) - float(mu)) ** 2 - float(mu)
        for y, mu in zip(actuals, predicted_means)
    )
    if denominator <= 0:
        return 1e6
    return max(0.5, min(1e6, numerator / denominator))


class PoissonRegression:
    """단일 타깃(홈 또는 원정 득점) 포아송 회귀. log-link, 의존성 없음."""

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES):
        self.feature_names = tuple(feature_names)
        self.means = [0.0] * len(self.feature_names)
        self.scales = [1.0] * len(self.feature_names)
        self.weights = [0.0] * len(self.feature_names)
        self.intercept = 1.35  # log(league-average runs~3.8/9innings-ish) 근처에서 시작

    def _vector(self, row: dict[str, float]) -> list[float]:
        return [float(row.get(name, 0.0)) for name in self.feature_names]

    def _standardize(self, vector: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(vector, self.means, self.scales)]

    def _lambda(self, standardized: Sequence[float]) -> float:
        eta = self.intercept + sum(w * v for w, v in zip(self.weights, standardized))
        eta = max(-4.0, min(4.0, eta))  # exp 폭주 방지 (0.02~55점 범위로 제한)
        return math.exp(eta)

    def fit(
        self,
        rows: Sequence[dict[str, float]],
        runs: Sequence[int],
        epochs: int = 2000,
        learning_rate: float = 0.02,
        l2: float = 0.02,
    ) -> "PoissonRegression":
        if len(rows) != len(runs) or len(rows) < 20:
            raise ValueError("At least 20 aligned historical games are required")
        if any(not math.isfinite(float(v)) or v < 0 or int(v) != v for v in runs):
            raise ValueError("Run labels must be finite nonnegative integers")
        matrix = [self._vector(row) for row in rows]
        n = len(matrix)
        for j in range(len(self.feature_names)):
            values = [row[j] for row in matrix]
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            self.means[j] = mean
            self.scales[j] = max(1e-6, math.sqrt(variance))
        x = [self._standardize(row) for row in matrix]
        # Optional acceleration preserves the same gradient updates for large
        # historical datasets. The dependency-free path remains available.
        if n >= 1000:
            try:
                import numpy as np
            except ImportError:
                pass
            else:
                design = np.asarray(x, dtype=float)
                targets = np.asarray(runs, dtype=float)
                weights = np.asarray(self.weights, dtype=float)
                for _ in range(epochs):
                    errors = np.exp(np.clip(self.intercept + design @ weights, -4, 4)) - targets
                    self.intercept -= learning_rate * float(errors.mean())
                    weights -= learning_rate * (design.T @ errors / n + l2 * weights)
                self.weights = weights.tolist()
                return self
        for _ in range(epochs):
            grad_w = [0.0] * len(self.weights)
            grad_b = 0.0
            for vector, target in zip(x, runs):
                lam = self._lambda(vector)
                error = lam - int(target)  # 포아송 로그가능도 그래디언트: (lambda - y)
                grad_b += error
                for j, value in enumerate(vector):
                    grad_w[j] += error * value
            self.intercept -= learning_rate * grad_b / n
            for j in range(len(self.weights)):
                penalty = l2 * self.weights[j]
                self.weights[j] -= learning_rate * (grad_w[j] / n + penalty)
        return self

    def predict_lambda(self, features: dict[str, float]) -> float:
        return self._lambda(self._standardize(self._vector(features)))

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "means": self.means, "scales": self.scales,
            "weights": self.weights, "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PoissonRegression":
        model = cls(tuple(payload["feature_names"]))
        model.means = payload["means"]; model.scales = payload["scales"]
        model.weights = payload["weights"]; model.intercept = payload["intercept"]
        return model


class TeamScoreModel:
    """홈·원정 평균 회귀와 NB 과산포를 결합한 점수분포 모델."""

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES, *, segment: str = "full"):
        if segment not in {"full", "f5"}:
            raise ValueError("segment must be full or f5")
        self.segment = segment
        self.home = PoissonRegression(feature_names)
        self.away = PoissonRegression(feature_names)
        self.feature_names = tuple(feature_names)
        self.home_dispersion = 1e6
        self.away_dispersion = 1e6
        self.model_kind = "score_distribution_nb_v1"

    def fit(self, rows: Sequence[dict[str, float]], home_runs: Sequence[int], away_runs: Sequence[int]) -> "TeamScoreModel":
        self.home.fit(rows, home_runs)
        self.away.fit(rows, away_runs)
        self.home_dispersion = estimate_dispersion(
            home_runs, [self.home.predict_lambda(row) for row in rows]
        )
        self.away_dispersion = estimate_dispersion(
            away_runs, [self.away.predict_lambda(row) for row in rows]
        )
        return self

    def score_grid(self, features: dict[str, float], max_runs: int = MAX_RUNS) -> list[list[float]]:
        """grid[a][h] = P(away_runs=a, home_runs=h). 홈·원정 독립 가정(baseline)."""
        home_lambda = self.home.predict_lambda(features)
        away_lambda = self.away.predict_lambda(features)
        if not all(math.isfinite(v) for v in (home_lambda, away_lambda)):
            raise ValueError("Non-finite score prediction")
        max_runs = max(max_runs, int(max(home_lambda, away_lambda)))
        def marginal_mass(mu: float, dispersion: float, limit: int) -> float:
            return sum(_nb_pmf(k, mu, dispersion) for k in range(limit + 1))

        while min(
            marginal_mass(home_lambda, self.home_dispersion, max_runs),
            marginal_mass(away_lambda, self.away_dispersion, max_runs),
        ) < 1 - 1e-10:
            max_runs += 1
            if max_runs > 512:
                raise ValueError("Score-distribution tail did not converge")
        home_pmf = [
            _nb_pmf(h, home_lambda, self.home_dispersion)
            for h in range(max_runs + 1)
        ]
        away_pmf = [
            _nb_pmf(a, away_lambda, self.away_dispersion)
            for a in range(max_runs + 1)
        ]
        mass = sum(home_pmf) * sum(away_pmf)
        return [[a * h / mass for h in home_pmf] for a in away_pmf]

    def save(self, path: str | Path) -> None:
        payload = {
            "model_kind": self.model_kind,
            "feature_version": FEATURE_VERSION,
            "segment": self.segment,
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "home_dispersion": self.home_dispersion,
            "away_dispersion": self.away_dispersion,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TeamScoreModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Score model uses feature_version {payload.get('feature_version')!r}, "
                f"expected {FEATURE_VERSION}. Retrain."
            )
        for side in ("home", "away"):
            values = payload[side]
            if tuple(values.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError("Score model feature order does not match v2.0")
            for field in ("weights", "means", "scales"):
                if len(values[field]) != len(FEATURE_NAMES) or not all(math.isfinite(float(v)) for v in values[field]):
                    raise ValueError(f"Invalid model {field}")
            if any(v <= 0 for v in values["scales"]) or not math.isfinite(values["intercept"]):
                raise ValueError("Invalid model scales/intercept")
        model_kind = payload.get("model_kind", "score_distribution_v1")
        if model_kind not in {"score_distribution_v1", "score_distribution_nb_v1"}:
            raise ValueError(f"Unsupported score model kind: {model_kind}")
        model = cls(segment=payload.get("segment", "full"))
        model.home = PoissonRegression.from_dict(payload["home"])
        model.away = PoissonRegression.from_dict(payload["away"])
        model.model_kind = model_kind
        model.home_dispersion = float(payload.get("home_dispersion", 1e6))
        model.away_dispersion = float(payload.get("away_dispersion", 1e6))
        if not all(
            math.isfinite(value) and value > 0
            for value in (model.home_dispersion, model.away_dispersion)
        ):
            raise ValueError("Invalid score-model dispersion")
        return model


# ---- 그리드 -> 마켓 확률 변환 (정산 규칙이 명확한 3개 마켓만) ----

def moneyline_probability(grid: list[list[float]], selection: str) -> float:
    """selection: '승'(홈 승) 또는 '패'(원정 승). 동점(연장으로 감)은 그리드에서 제외."""
    if selection not in {"승", "패"}:
        raise ValueError("Unsupported moneyline selection")
    home_win = sum(grid[a][h] for a in range(len(grid)) for h in range(len(grid[a])) if h > a)
    away_win = sum(grid[a][h] for a in range(len(grid)) for h in range(len(grid[a])) if a > h)
    decided = home_win + away_win
    if decided <= 0:
        return 0.5
    return (home_win if selection == "승" else away_win) / decided  # 무승부 제외 재정규화


def total_probability(grid: list[list[float]], selection: str, line: float) -> float:
    """selection: '언더' 또는 '오버'. line은 .5로 끝나는 값(push 없음 가정)."""
    if selection not in {"언더", "오버"} or not math.isfinite(line) or line % 1 != 0.5:
        raise ValueError("Total requires 언더/오버 and a half-run line; push rules are not configured")
    return sum(grid[a][h] for a in range(len(grid)) for h in range(len(grid[a]))
               if (a + h < line if selection == "언더" else a + h > line))


def handicap_probability(grid: list[list[float]], selection: str, line: float) -> float:
    """selection: '승'(홈+line) 또는 '패'(원정 그대로). line은 홈 기준 부호(+2.5, -1.5 등)."""
    if selection not in {"승", "패"} or not math.isfinite(line) or line % 1 != 0.5:
        raise ValueError("Handicap requires 승/패 and a half-run line; push rules are not configured")
    home_covers = sum(
        grid[a][h] for a in range(len(grid)) for h in range(len(grid[a])) if h + line > a
    )
    mass = sum(map(sum, grid))
    return home_covers if selection == "승" else mass - home_covers


def win_one_loss_probability(grid: list[list[float]], selection: str) -> float:
    """프로토 야구 승1패: 홈 2점+ 승 / 1점 차 이내 / 홈 2점+ 패."""
    normalized = "1" if selection in {"1", "①", "무"} else selection
    if normalized not in {"승", "1", "패"}:
        raise ValueError("Unsupported 승1패 selection")
    return sum(
        grid[a][h]
        for a in range(len(grid))
        for h in range(len(grid[a]))
        if (
            h - a >= 2
            if normalized == "승"
            else a - h >= 2
            if normalized == "패"
            else abs(h - a) <= 1
        )
    )


def three_way_probability(grid: list[list[float]], selection: str) -> float:
    """승무패: 홈 승 / 동점 / 홈 패. 전반 5이닝 마켓에 사용한다."""
    if selection not in {"승", "무", "패"}:
        raise ValueError("Unsupported three-way selection")
    return sum(
        grid[a][h]
        for a in range(len(grid))
        for h in range(len(grid[a]))
        if (h > a if selection == "승" else h == a if selection == "무" else h < a)
    )


def sum_probability(grid: list[list[float]], selection: str) -> float:
    """프로토 SUM: 양 팀 최종 득점 합계의 홀/짝."""
    if selection not in {"홀", "짝"}:
        raise ValueError("Unsupported SUM selection")
    parity = 1 if selection == "홀" else 0
    return sum(
        grid[a][h]
        for a in range(len(grid))
        for h in range(len(grid[a]))
        if (a + h) % 2 == parity
    )
