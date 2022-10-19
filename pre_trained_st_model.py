from typing import Dict, Union

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from model_stacker import ModelStack


class HeadModel:
    def __init__(self, model_class, *model_args):
        self.model = model_class(*model_args)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class SentenceTransformerHeadModel:
    def __init__(
        self, model_path: str, head_model: HeadModel, *head_model_args
    ) -> None:
        self.model = SentenceTransformer(model_path)
        self.head_model = head_model(*head_model_args)

    def fit(self, X_train, y_train):
        print("Encoding training set")
        X_train_embeddings = self.model.encode(X_train)
        print("Fitting Ridge CV...")
        self.head_model.fit(X_train_embeddings, y_train)

    def score(self, X_test, y_test):
        print("Encoding test set")
        X_test_embeddings = self.model.encode(X_test)
        print("Scoring...")
        score = self.head_model.score(X_test_embeddings, y_test)
        return score

    def predict(self, X_test):
        print("Encoding test set")
        X_test_embeddings = self.model.encode(X_test)
        print("Predicting...")
        predictions = self.head_model.predict(X_test_embeddings)
        return predictions


class SentenceTransformerModelRidgeCV:
    def __init__(self, model_path: str) -> None:
        self.model = SentenceTransformer(model_path)
        super().__init__(model_path, RidgeCV)


class MultiHeadSentenceTransformerModel:
    def __init__(
        self,
        model: Union[str, SentenceTransformer, "ModelStack"],
        head_model: HeadModel,
        *head_model_args,
        **head_model_kwargs,
    ) -> None:
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        elif isinstance(model, SentenceTransformer):
            self.model = model
        elif isinstance(model, ModelStack):
            self.model = model
        else:
            raise ValueError("Invalid model type")
        self.heads = {}
        self.head_model = head_model
        self.head_model_args = head_model_args
        self.head_model_kwargs = head_model_kwargs

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
        print(f"Fitting {self.head_model.__class__.__name__} on {attribute} ...")
        self.heads[attribute] = self.head_model(
            *self.head_model_args, **self.head_model_kwargs
        )
        self.heads[attribute].fit(X_train, y_train)

    def score(self, attribute, X_test, y_test):
        score = self.heads[attribute].score(X_test, y_test)
        return score

    def predict(self, attribute, X_test):
        predictions = self.heads[attribute].predict(X_test)
        return predictions


class MultiHeadSentenceTransformerModelRidgeCV(MultiHeadSentenceTransformerModel):
    def __init__(self, model: Union[str, SentenceTransformer, "ModelStack"]) -> None:
        super().__init__(model, RidgeCV)


class MultiHeadSentenceTransformerFactory:
    @staticmethod
    def create_class(head_model: HeadModel, *head_args, **head_model_kwargs):
        class MultiHeadSentenceTransformerFromFactory(
            MultiHeadSentenceTransformerModel
        ):
            def __init__(
                self, model: Union[str, SentenceTransformer, "ModelStack"]
            ) -> None:
                super().__init__(model, head_model, *head_args, **head_model_kwargs)

        return MultiHeadSentenceTransformerFromFactory


class MultiClassMultiHeadSentenceTransformerModel(MultiHeadSentenceTransformerModel):
    def __init__(self, model: Union[str, SentenceTransformer, "ModelStack"]):
        super().__init__(model, None, None)
        self.heads = {}
        self.use_scalers = {}
        self.scalers = {}

    def add_head(
        self, attribute, use_scaler, head_model, *head_args, **head_model_kwargs
    ):
        print(head_model_kwargs)
        self.heads[attribute] = head_model(*head_args, **head_model_kwargs)
        self.use_scalers[attribute] = use_scaler

    def fit(self, attribute, X_train, y_train):
        if attribute not in self.heads:
            TypeError(
                f"You must add a new head to the model for the attribute: {attribute}, before you can fit it."
            )
        if self.use_scalers[attribute]:
            self.scalers[attribute] = StandardScaler()
            X_train = self.scalers[attribute].fit_transform(X_train)

        print(f"Fitting {self.heads[attribute]} on {attribute} ...")
        self.heads[attribute].fit(X_train, y_train)

    def score(self, attribute, X_test, y_test):
        if self.use_scalers[attribute]:
            X_test = self.scalers[attribute].transform(X_test)

        score = self.heads[attribute].score(X_test, y_test)
        return score

    def predict(self, attribute, X_test):
        if self.use_scalers[attribute]:
            X_test = self.scalers[attribute].transform(X_test)

        predictions = self.heads[attribute].predict(X_test)
        return predictions


class MultiEncodingStack:
    def __init__(self) -> None:
        self.encoding_stacks = {}
        self.regressor_heads = {}

    def add_encoding_stack(self, attribute, stack: ModelStack):
        self.encoding_stacks[attribute] = stack

    def add_head(self, attribute, head_model, *head_args, **head_kwargs):
        self.regressor_heads[attribute] = head_model(*head_args, **head_kwargs)

    def encode(self, attribute, X, **kwargs):
        return self.encoding_stacks[attribute].encode(X, **kwargs)

    def fit(self, attribute, X_train, y_train):
        self.regressor_heads[attribute].fit(X_train, y_train)

    def score(self, attribute, X_test, y_test):
        score = self.regressor_heads[attribute].score(X_test, y_test)
        return score

    def predict(self, attribute, X_test):
        predictions = self.regressor_heads[attribute].predict(X_test)
        return predictions
