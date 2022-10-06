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

class MCRMSECalculator:
    def __init__(self):
        self._sum = 0.0
        self._samples = 0

    def compute_column(self, labels, predictions):
        points = zip(labels, predictions)
        column_sum = 0.0
        for point in points:
            column_sum += (point[0] - point[1]) ** 2
        self._sum += column_sum / len(labels)
        self._samples += 1

    def get_score(self):
        return self._sum / self._samples
