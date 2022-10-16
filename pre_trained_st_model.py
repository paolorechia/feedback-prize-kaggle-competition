from sklearn.linear_model import RidgeCV
from sentence_transformers import SentenceTransformer
from utils import attributes


class SentenceTransformerModelRidgeCV:
    def __init__(self, model_path: str) -> None:
        self.model = SentenceTransformer(model_path)
        self.ridge_cv = RidgeCV()

    def fit(self, X_train, y_train):
        print("Encoding training set")
        X_train_embeddings = self.model.encode(X_train)
        print("Fitting Ridge CV...")
        self.ridge_cv.fit(X_train_embeddings, y_train)

    def score(self, X_test, y_test):
        print("Encoding test set")
        X_test_embeddings = self.model.encode(X_test)
        print("Scoring...")
        score = self.ridge_cv.score(X_test_embeddings, y_test)
        return score

    def predict(self, X_test):
        print("Encoding test set")
        X_test_embeddings = self.model.encode(X_test)
        print("Predicting...")
        predictions = self.ridge_cv.predict(X_test_embeddings)
        return predictions


class MultiHeadSentenceTransformerModelRidgeCV:
    def __init__(self, model_path: str) -> None:
        self.model = SentenceTransformer(model_path)
        self.heads = {k: RidgeCV() for k in attributes}

    def fit(self, attribute, X_train, y_train):
        print("Encoding training set")
        X_train_embeddings = self.model.encode(X_train)
        print("Fitting Ridge CV...")
        self.heads[attribute].fit(X_train_embeddings, y_train)

    def score(self, attribute, X_test, y_test):
        print("Encoding test set")
        X_test_embeddings = self.model.encode(X_test)
        print("Scoring...")
        score = self.heads[attribute].score(X_test_embeddings, y_test)
        return score

    def predict(self, attribute, X_test):
        print("Encoding test set")
        X_test_embeddings = self.model.encode(X_test)
        print("Predicting...")
        predictions = self.heads[attribute].predict(X_test_embeddings)
        return predictions
