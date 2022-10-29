from typing import List

import pandas as pd

from pre_trained_st_model import (
    MultiBlockRidgeCV,
    fit_multi_block,
    predict_multi_block,
    score_multi_block,
)


class Fold:
    def __init__(
        self,
        fold_number: int,
        train_df: pd.DataFrame,
        train_indices,
        y_train,
        y_trains,
        train_text_ids,
        test_df: pd.DataFrame,
        test_indices,
        y_test,
    ):
        self.fold_number = fold_number

        self.train_df = train_df
        self.train_indices = train_indices

        self.test_df = test_df
        self.test_indices = test_indices

        self.y_train = y_train
        self.train_text_ids = train_text_ids
        self.y_trains = y_trains
        self.y_test = y_test
        self.y_trains_new = y_trains.copy()
        self.id_idx_mapping = {}

        for idx, text_id in enumerate(train_text_ids):
            self.id_idx_mapping[text_id] = idx

    def is_text_in_train_fold(self, text_id: str) -> bool:
        return text_id in self.train_text_ids

    def replace_label(self, text_id: str, new_labels: List[float]):
        if text_id not in self.id_idx_mapping:
            raise ValueError(f"Text id {text_id} not found in train set")
        idx = self.id_idx_mapping[text_id]
        for j, label in enumerate(new_labels):
            self.y_trains_new[j][idx] = label

    def save_labels(self):
        self.y_trains = self.y_trains_new.copy()

    def restore_labels(self):
        self.y_trains_new = self.y_trains.copy()

    def __str__(self):
        return f"Fold {self.fold_number} (train: {len(self.train_df)}, test: {len(self.test_df)})"

    def __repr__(self):
        return str(self)


def fit_fold(
    fold: Fold,
    multi_block: MultiBlockRidgeCV,
    attribute: str,
    averager_regressor,
):
    fit_multi_block(
        multi_block,
        attribute,
        fold.train_df,
        fold.train_indices,
        fold.y_train,
        fold.y_trains_new,
        averager_regressor,
    )


def score_fold(
    fold: Fold,
    multi_block: MultiBlockRidgeCV,
    attribute: str,
    averager_regressor,
    average_function,
    weights: List[float],
):
    y_pred = predict_multi_block(
        multi_block, attribute, fold.test_df, fold.test_indices
    )
    return score_multi_block(
        fold.y_test,
        y_pred,
        averager_regressor,
        average_function,
        weights,
    )
