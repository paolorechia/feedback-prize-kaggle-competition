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


def run(train_size, dataset_size):
    # Load the model
    val_size = 1 - train_size
    test_df, val_df = create_train_test_df(train_size, "full")

    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    degradation_rate = 0.1
    epochs = 0
    use_fine_tuning = False

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
    test_df["full_text"] = test_df["full_text"].apply(remove_repeated_whitespaces)
    X_test = multi_block.encode(
        test_df["full_text"], cache_type=f"no-ws-full-{train_size}"
    )
    test_df["embeddings_0"] = [np.array(e) for e in X_test]
    test_indices = test_df.index.values

    val_df["full_text"] = val_df["full_text"].apply(remove_repeated_whitespaces)
    X_val = multi_block.encode(val_df["full_text"], cache_type=f"no-ws-val-{val_size}")
    val_df["embeddings_0"] = [np.array(e) for e in X_val]
    val_indices = val_df.index.values

    train_df_cache_key = (
        f"train-{train_size}-dss-{dataset_size}-degradation-{degradation_rate}"
    )

    used_datasets = [
        ("bbc", load_bbc_news, True),
        ("amazon", load_amazon_reviews, True),
        ("steam", load_steam_reviews, True),
        ("goodreads", load_goodreads_reviews, True),
    ]
    for name, _, _ in used_datasets:
        train_df_cache_key += f"-{name}"

    train_csv_path = f"{train_df_cache_key}.csv"
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
    else:
        texts = []
        for name, load_func, is_used in used_datasets:
            if not is_used:
                continue

            text = load_func(head=dataset_size - 1)
            texts.extend(text)

        train_df = pd.DataFrame(texts)
        train_df["review_text"] = train_df["review_text"].apply(
            remove_repeated_whitespaces
        )
        train_df = degradate_df_text(
            train_df, "review_text", degradation_rate=degradation_rate
        )
        train_df.to_csv(train_csv_path, index=False)

    print(train_df.review_text.head())
    print(len(train_df))

    X_train = multi_block.encode(
        train_df["review_text"],
        cache_type=train_df_cache_key,
    )

    train_df["embeddings_0"] = [np.array(e) for e in X_train]

    train_indices = train_df.index.values

    fined_tuned = {}
    fine_tuning_interval = 10
    for attribute in attributes:
        print("Attribute", attribute)
        y_test = np.array(test_df[attribute])
        y_val = np.array(val_df[attribute])

        initial_fit_df = test_df

        y_initial = np.array(initial_fit_df[attribute])
        y_initials = [y_initial]

        fit_multi_block(
            multi_block,
            attribute,
            initial_fit_df,
            initial_fit_df.index.values,
            y_initial,
            y_initials,
        )

        first_pred = predict_multi_block(
            multi_block, attribute, train_df, train_indices
        )
        first_pred = first_pred[0]
        fit_multi_block(
            multi_block,
            attribute,
            train_df,
            train_indices,
            first_pred,
            [first_pred],
        )
        second_pred = predict_multi_block(multi_block, attribute, test_df, test_indices)
        second_pred = second_pred[0]

        initial_score = calculate_rmse_score_single(y_test, second_pred)
        print("Initial score: ", initial_score)
        best = second_pred
        previous_score = 100.00
        for i in range(epochs):
            new_initial_labels = predict_multi_block(
                multi_block, attribute, initial_fit_df, initial_fit_df.index.values
            )
            fit_multi_block(
                multi_block,
                attribute,
                initial_fit_df,
                initial_fit_df.index.values,
                new_initial_labels[0],
                new_initial_labels,
            )
            first_pred = predict_multi_block(
                multi_block, attribute, train_df, train_indices
            )

            first_pred = first_pred[0]
            fit_multi_block(
                multi_block,
                attribute,
                train_df,
                train_indices,
                first_pred,
                [first_pred],
            )
            second_pred = predict_multi_block(
                multi_block, attribute, test_df, test_indices
            )
            second_pred = second_pred[0]
            best = second_pred
            score = calculate_rmse_score_single(y_test, second_pred)
            print("Second pred score: ", score)

            # if score > previous_score:
            #     break
            previous_score = score

        train_df[attribute] = [
            fit_float_score_to_nearest_valid_point(x) for x in first_pred
        ]
        # train_df.rename(columns={"review_text": "full_text"}, inplace=True)
        train_df["full_text"] = train_df["review_text"]
        train_df["text_id"] = train_df.index.values
        train_df.drop(columns=["embeddings_0"]).to_csv(
            train_df_cache_key + ".csv", index=False
        )
        print("Saved to", train_df_cache_key + ".csv")

        # Begin label fine tuning
        if use_fine_tuning:
            print("Begin label fine tuning")
            y_train = np.array(train_df[attribute])
            print("Len(y_train)", len(y_train))
            best_score = previous_score

            for idx, _ in train_df.iterrows():
                if idx % fine_tuning_interval != 0:
                    continue
                print("Idx: ", idx)

                best_label = y_train[idx]
                for label in possible_labels:
                    y_train[idx] = label
                    fit_multi_block(
                        multi_block,
                        attribute,
                        train_df,
                        train_indices,
                        y_train,
                        [y_train],
                    )
                    second_pred = predict_multi_block(
                        multi_block, attribute, test_df, test_indices
                    )
                    second_pred = second_pred[0]
                    score = calculate_rmse_score_single(y_test, second_pred)
                    print("Fine tuning score: ", score)
                    if score < best_score:
                        print("New best score: ", score)
                        best_label = label
                        best_score = score
                y_train[idx] = best_label
                fined_tuned[attribute] = y_train

    if use_fine_tuning:
        for attribute in attributes:
            train_df[attribute] = fined_tuned[attribute]
        train_df[attribute] = y_train
        train_df.drop(columns=["embeddings_0"]).to_csv(
            train_df_cache_key + f"_fine_tuned_{fine_tuning_interval}" + ".csv",
            index=False,
        )
        print(
            "Saved to",
            train_df_cache_key + f"_fine_tuned_{fine_tuning_interval}" + ".csv",
        )


if __name__ == "__main__":
    dataset_size = 100000
    # for ts in [0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275]:
        # run(ts)
    run(0.2, dataset_size)