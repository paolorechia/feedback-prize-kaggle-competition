"Small helper to load an attribute fold"
from typing import Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os

full_df_path = "/data/feedback-prize/train.csv"
sampled_df_path = "./small_sets/full_sampled_set.csv"
split_csv_dirs = "./split_csvs"
intermediate_df_path = "/data/feedback-prize/intermediate.csv"
fold_df_path = "/data/feedback-prize/"
text_label = "full_text"
random_state = 10


def create_attribute_stratified_split(
    attribute: str, test_size: float, dataset: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "full":
        full_df = pd.read_csv(full_df_path)
    elif dataset == "sampled":
        full_df = pd.read_csv(sampled_df_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    split_train_df_path = f"{split_csv_dirs}/train_{attribute}_{test_size}.csv"
    split_test_df_path = f"{split_csv_dirs}/test_{attribute}_{test_size}.csv"

    # Uses pre-generated files if they exist
    if os.path.exists(split_train_df_path) and os.path.exists(split_test_df_path):
        train_df = pd.read_csv(split_train_df_path)
        test_df = pd.read_csv(split_test_df_path)
        return train_df, test_df

    text_label = "full_text"

    X = full_df[text_label]
    y = full_df[attribute]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_index, test_index = next(sss.split(X, y))

    train_df = full_df.filter(items=train_index, axis=0)
    test_df = full_df.filter(items=test_index, axis=0)

    train_df.to_csv(split_train_df_path, index=False)
    test_df.to_csv(split_test_df_path, index=False)

    return train_df, test_df


def sample_sentences_per_class(
    df: pd.DataFrame, attribute: str, max_samples_per_class: int
) -> pd.DataFrame:
    labels = df[attribute].unique()
    small_subset = pd.DataFrame()

    for label in labels:
        label_df = df[df[attribute] == label]
        print(label, len(label_df))
        small_subset = pd.concat(
            [small_subset, label_df.sample(min(max_samples_per_class, len(label_df)))]
        )
    return small_subset
