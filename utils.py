from warnings import warn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

attributes = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]


def fit_float_score_to_nearest_valid_point(float_score: float):
    """Fit float score to nearest valid point."""
    valid_points = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    return min(valid_points, key=lambda x: abs(x - float_score))


def round_border_score(float_score: float):
    if float_score < 1.0:
        return 1.0
    if float_score > 5.0:
        return 5.0
    return float_score


def test_fit_float_score_to_nearest_valid_point():
    """Tests function"""
    assert fit_float_score_to_nearest_valid_point(0.0) == 1.0
    assert fit_float_score_to_nearest_valid_point(0.9) == 1.0
    assert fit_float_score_to_nearest_valid_point(1.1) == 1.0
    assert fit_float_score_to_nearest_valid_point(1.24) == 1.0
    assert fit_float_score_to_nearest_valid_point(1.27) == 1.5
    assert fit_float_score_to_nearest_valid_point(1.74) == 1.5
    assert fit_float_score_to_nearest_valid_point(1.76) == 2.0
    assert fit_float_score_to_nearest_valid_point(2.26) == 2.5
    assert fit_float_score_to_nearest_valid_point(2.76) == 3.0
    assert fit_float_score_to_nearest_valid_point(3.3) == 3.5
    assert fit_float_score_to_nearest_valid_point(3.6) == 3.5
    assert fit_float_score_to_nearest_valid_point(3.76) == 4.0
    assert fit_float_score_to_nearest_valid_point(4.2) == 4.0
    assert fit_float_score_to_nearest_valid_point(4.3) == 4.5
    assert fit_float_score_to_nearest_valid_point(4.6) == 4.5
    assert fit_float_score_to_nearest_valid_point(4.76) == 5.0
    assert fit_float_score_to_nearest_valid_point(5.0) == 5.0
    assert fit_float_score_to_nearest_valid_point(10.0) == 5.0


def calculate_rmse_score(y_true, y_pred):
    rmse_scores = []
    for i in range(len(attributes)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    return np.mean(rmse_scores)


def calculate_rmse_score_single(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_rmse_score_attribute(attribute, y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true[attribute], y_pred[attribute]))
