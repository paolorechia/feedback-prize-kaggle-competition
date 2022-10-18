from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from model_catalog import Model
from model_loader import load_model_with_dropout


class StackedModel:
    def __init__(self, model_info: Model, st: SentenceTransformer) -> None:
        self.info: Model = model_info
        self.model: SentenceTransformer = st

    def encode(
        self,
        text: Union[List[str], str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy=True,
    ) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        return self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
        )


class ModelStack:
    def __init__(self, models: List[Model]):
        self.stack = []
        for model in models:
            st_model = load_model_with_dropout(
                model, attention_dropout=0.0, hidden_dropout=0.0
            )
            self.stack.append(StackedModel(model, st_model))

    def encode(
        self,
        text: Union[List[str], str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_numpy=True,
    ) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        embeddings = []
        for model in self.stack:
            embeddings_ = model.encode(
                text,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
            )
            print("embeddings_.shape", embeddings_.shape)
            embeddings.append(embeddings_)

        encoded = np.concatenate(embeddings, axis=1)
        print(f"Encoded shape: {encoded.shape}")
        return encoded
