attributes = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]


labels = {
    "1.0": "terrible",
    "1.5": "bad",
    "2.0": "poor",
    "2.5": "fair",
    "3.0": "average",
    "3.5": "good",
    "4.0": "great",
    "4.5": "excellent",
    "5.0": "perfect",
}
reverse_labels = {v: float(k) for k, v in labels.items()}


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


class MCRMSECalculator:
    def __init__(self):
        self._sum = 0.0
        self._samples = 0

    def compute_columns(self, class_labels, predictions):
        """Need to fix this function"""
        for idx, labels in enumerate(class_labels):
            preds = predictions[idx]
            self.compute_column(labels, preds)
        self._samples = len(class_labels)

    def compute_column(self, labels, predictions):
        points = zip(labels, predictions)
        column_sum = 0.0
        for point in points:
            column_sum += (point[0] - point[1]) ** 2
        self._sum += column_sum / len(labels)
        self._samples += 1

    def get_score(self):
        return self._sum / self._samples
