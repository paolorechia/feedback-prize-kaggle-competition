from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from model_catalog import Model
from model_loader import load_model_with_dropout

import os

cache_encodings_dir = "/data/cache_encodings"


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
        use_cache=True,
        cache_type="train",
    ) -> np.ndarray:
        if isinstance(text, str):
            text = [text]

        if use_cache:
            if not os.path.exists(cache_encodings_dir):
                os.makedirs(cache_encodings_dir)

            cache_file = os.path.join(
                cache_encodings_dir,
                self.info.model_name.replace("/", "_") + f"_{cache_type}" + ".npy",
            )

            if os.path.exists(cache_file):
                return np.load(cache_file)

        array = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
        )
        if use_cache:
            np.save(cache_file, array)
        return array


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
        use_cache=True,
        cache_type="train",
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
                use_cache=use_cache,
                cache_type=cache_type,
            )
            print("embeddings_.shape", embeddings_.shape)
            embeddings.append(embeddings_)

        encoded = np.concatenate(embeddings, axis=1)
        print(f"Encoded shape: {encoded.shape}")
        return encoded
