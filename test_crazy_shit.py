import numpy as np
import pandas as pd
from load_data import create_train_test_df
import os

from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import (
    MultiBlockRidgeCV,
    MultiBlockSGD,
    MultiBlockRidge,
    MultiBlockBayensianRidge,
    predict_multi_block,
    fit_multi_block,
)
from utils import (
    attributes,
    calculate_rmse_score_single,
    fit_float_score_to_nearest_valid_point,
    possible_labels,
    remove_repeated_whitespaces,
)
from text_degradation import degradate_df_text
from load_extra_datasets import (
    load_steam_reviews,
    load_bbc_news,
    load_amazon_reviews,
    load_goodreads_reviews,
)
import warnings

warnings.filterwarnings("ignore")


def main():
    # Load the model
    model_info = ModelCatalog.DebertaV3
    multi_block_class = MultiBlockRidgeCV
    multi_block = multi_block_class(
        model=ModelStack(
            [
                model_info,
            ],
        ),
        number_blocks=1,
        labels=attributes,
    )

    # Load the data
    train_size = 0.2
    val_size = 1 - train_size
    fine_tuning_interval = 10
    test_df, val_df = create_train_test_df(train_size, "full")

    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    use_fine_tuning = False

    augmented_csv = "train-1000-degradation-0.1-bbc-amazon-steam-goodreads"
    if use_fine_tuning:
        augmented_csv += f"_fine_tuned_{fine_tuning_interval}"
    augmented_csv += ".csv"
    augmented_df = pd.read_csv(augmented_csv)

    if use_fine_tuning:
        N = pd.read_csv("train-1000-degradation-0.1-bbc-amazon-steam-goodreads.csv")
        A = pd.read_csv(augmented_csv)

        for attribute in attributes:
            print(attribute)
            print("normal, fine_tuned")
            equals = 0
            different = 0
            for n, a in zip(N[attribute], A[attribute]):
                if n == a:
                    equals += 1
                else:
                    different += 1
            print("Equals: ", equals)
            print("Different: ", different)

    used_csvs = [
        # ("test", test_df),
        ("augmented", augmented_df),
        # Keep this like this, please
    ]

    cache_suffix = "-".join([t[0] for t in used_csvs])
    cache_key = (
        f"test-crazy-fine-tuned-{augmented_csv}-{cache_suffix}"
        if use_fine_tuning
        else f"test-crazy-shit-{augmented_csv}-{cache_suffix}"
    )

    dfs = []
    for _, df in used_csvs:
        dfs.append(df)

    train_df = pd.concat(dfs, ignore_index=True)
    train_df["full_text"] = train_df["full_text"].astype(str)
    train_df["full_text"] = train_df["full_text"].apply(remove_repeated_whitespaces)

    X_test = multi_block.encode(train_df["full_text"], cache_type=cache_key)
    train_df["embeddings_0"] = [np.array(e) for e in X_test]

    val_df["full_text"] = val_df["full_text"].apply(remove_repeated_whitespaces)

    X_val = multi_block.encode(
        val_df["full_text"], cache_type=f"val-crazy-shit-2-{augmented_csv}"
    )
    val_df["embeddings_0"] = [np.array(e) for e in X_val]

    scores = []
    for attribute in attributes:
        print("Attribute", attribute)
        y_train = np.array(train_df[attribute])
        y_val = np.array(val_df[attribute])
        fit_multi_block(
            multi_block,
            attribute,
            train_df,
            train_df.index.values,
            y_train,
            [y_train],
        )
        val_preds = predict_multi_block(
            multi_block, attribute, val_df, val_df.index.values
        )
        score = calculate_rmse_score_single(y_val, val_preds[0])
        print("Second pred score: ", score)
        scores.append(score)
    print("Mean score: ", np.mean(scores))


if __name__ == "__main__":
    main()
