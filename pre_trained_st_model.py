from typing import Union
from sklearn.linear_model import RidgeCV
from sentence_transformers import SentenceTransformer
from model_stacker import ModelStack


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
    def __init__(self, model: Union[str, SentenceTransformer, "ModelStack"]) -> None:
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        elif isinstance(model, SentenceTransformer):
            self.model = model
        elif isinstance(model, ModelStack):
            self.model = model
        else:
            raise ValueError("Invalid model type")
        self.heads = {}

    def encode(
        self,
        X,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        use_cache=True,
        type_path: str = "train",
    ):
        if isinstance(self.model, ModelStack):
            return self.model.encode(
                X,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                use_cache=use_cache,
                cache_type=type_path,
            )
        return self.model.encode(
            X,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
        )

    def fit(self, attribute, X_train, y_train):
        print("Fitting Ridge CV...")
        self.heads[attribute] = RidgeCV()
        self.heads[attribute].fit(X_train, y_train)

    def score(self, attribute, X_test, y_test):
        score = self.heads[attribute].score(X_test, y_test)
        return score

    def predict(self, attribute, X_test):
        print("Predicting...")
        predictions = self.heads[attribute].predict(X_test)
        return predictions
