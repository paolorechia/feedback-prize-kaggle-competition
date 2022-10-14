"Small helper to load an attribute fold"
from typing import Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

full_df_path = "/data/feedback-prize/train.csv"
split_csv_dirs = "./split_csvs"
intermediate_df_path = "/data/feedback-prize/intermediate.csv"
fold_df_path = "/data/feedback-prize/"
text_label = "full_text"
random_state = 10

full_df = pd.read_csv(full_df_path)

def create_attribute_stratified_split(attribute: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text_label = "full_text"

    X = full_df[text_label]
    y = full_df[attribute]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(sss.split(X, y))

    train_df = full_df.filter(items=train_index, axis=0)
    test_df = full_df.filter(items=test_index, axis=0)
    split_train_df_path = f"{split_csv_dirs}/train_{attribute}_{test_size}.csv"
    split_test_df_path = f"{split_csv_dirs}/test_{attribute}_{test_size}.csv"

    train_df.to_csv(split_train_df_path, index=False)
    test_df.to_csv(split_test_df_path, index=False)

    return train_df, test_df
