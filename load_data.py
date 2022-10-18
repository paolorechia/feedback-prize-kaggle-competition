"Small helper to load an attribute fold"
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from seeds import PANDAS_RANDOM_STATE
from utils import attributes

full_df_path = "/data/feedback-prize/train.csv"
sampled_df_path = "./small_sets/full_sampled_set.csv"
split_csv_dirs = "./split_csvs"
intermediate_df_path = "/data/feedback-prize/intermediate.csv"
fold_df_path = "/data/feedback-prize/"
text_label = "full_text"
random_state = PANDAS_RANDOM_STATE


def _read_csv(dataset: str):
    """Read a CSV file into a Pandas DataFrame.

    Args:
        dataset (str): The name of the dataset to read.

    Raises:
        ValueError: If the dataset name is unknown.

    Returns:
        Pandas DataFrame: The contents of the CSV file.
    """
    if dataset == "full":
        return pd.read_csv(full_df_path)
    elif dataset == "sampled":
        return pd.read_csv(sampled_df_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _load_pregenerated_csvs(
    dir_path: str, attribute: str, test_size: float
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:

    train_df_path = f"{dir_path}/train_{attribute}_{test_size}.csv"
    test_df_path = f"{dir_path}/test_{attribute}_{test_size}.csv"
    if os.path.exists(train_df_path) and os.path.exists(test_df_path):
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)
        return True, train_df, test_df

    return False, None, None


def create_train_test_df(
    test_size: float, dataset: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a stratified train and test split for the given dataset.
    If the split is already generated, it will be loaded from the disk.
    Otherwise, it will be generated and saved to the disk.

    Arguments:
        test_size: The size of the test split.
        dataset: The dataset to create the split for.

    Returns:
        A tuple of train and test dataframes.

    Raises:
        ValueError: If the given dataset is not supported.
    """

    is_available, _, test_df = _load_pregenerated_csvs(
        split_csv_dirs, "full", test_size
    )
    if is_available:
        return train_df, test_df
    # it is not yet generated, so generate it
    full_train_df_path = f"{split_csv_dirs}/train_full__{test_size}.csv"
    full_test_df_path = f"{split_csv_dirs}/test_full_{test_size}.csv"

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for attr in attributes:
        attr_train_df, attr_test_df = create_attribute_stratified_split(
            attr, test_size, dataset
        )
        test_df = pd.concat([test_df, attr_test_df])
        train_df = pd.concat([train_df, attr_train_df])

    train_df.drop_duplicates(subset=["text_id"], inplace=True)
    test_df.drop_duplicates(subset=["text_id"], inplace=True)

    train_df.to_csv(full_train_df_path, index=False)
    test_df.to_csv(full_test_df_path, index=False)

    return train_df, test_df


def create_attribute_stratified_split(
    attribute: str, test_size: float, dataset: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_available, train_df, test_df = _load_pregenerated_csvs(
        split_csv_dirs, attribute, test_size
    )
    if is_available:
        return train_df, test_df

    # it is not yet generated, so generate it
    full_df = _read_csv(dataset)

    split_train_df_path = f"{split_csv_dirs}/train_{attribute}_{test_size}.csv"
    split_test_df_path = f"{split_csv_dirs}/test_{attribute}_{test_size}.csv"

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
    """
    This function takes a dataframe and an attribute and returns a dataframe with a maximum of max_samples_per_class
    rows per class.

    :param df: The dataframe to be sampled
    :param attribute: The attribute to be used for sampling
    :param max_samples_per_class: The maximum number of rows per class
    :return: A dataframe with a maximum of max_samples_per_class rows per class
    """
    labels = df[attribute].unique()
    small_subset = pd.DataFrame()

    for label in labels:
        label_df = df[df[attribute] == label]
        # print(label, len(label_df))
        small_subset = pd.concat(
            [small_subset, label_df.sample(min(max_samples_per_class, len(label_df)))]
        )
    return small_subset
