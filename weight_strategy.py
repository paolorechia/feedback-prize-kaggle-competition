def average_function(preds, weights):
    sum_ = 0.0
    denonimator = sum(weights)
    for idx, p in enumerate(preds):
        sum_ += weights[idx] * p
    return sum_ / denonimator


class WeightingStrategy:
    def linear(n):
        return [i / n for i in range(n)]

    def lasso_cv(*args, **kwargs):
        return []

    def diminishing(n):
        return [1 / (i + 1) for i in range(n)]

    def diminishing_2(n):
        d = [1 / (i + 1) ** 2 for i in range(n)]
        d[0] = 0.5
        d[1] = 0.3
        d[2] = 0.2
        return d

    def step_decrease(n):
        return [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01][:n]

    def uniform(n):
        return [1 / n for _ in range(n)]

    def custom(*args):
        def c(_):
            l = []
            for i in args:
                l.append(i)
            return l

        return c

    def linear_net(*args):
        return []
