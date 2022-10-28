import logging
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import MultiBlockRidgeCV
from seeds import KFOLD_RANDOM_STATE
from splitter import (
    SplittingStrategy,
    smart_blockenizer,
    split_text_into_n_parts,
    split_text_into_sliding_windows,
)
from utils import attributes, calculate_rmse_score_single
from sklearn.linear_model import LassoCV
from my_nets import LinearNet


def fit_multi_block(
    multi_block, attribute, train_df, train, y_train, averager_regressor=None
):
    for i in range(multi_block.number_blocks):
        embeddings = np.array(list(train_df[f"embeddings_{i}"])).reshape(len(train), -1)
        multi_block.fit(i, attribute, embeddings, y_train)

    train_preds = []
    for i in range(multi_block.number_blocks):
        embeddings = np.array(list(train_df[f"embeddings_{i}"])).reshape(len(train), -1)
        train_preds.append(multi_block.predict(i, attribute, embeddings))

    if averager_regressor is not None:
        train_preds = np.array(train_preds).transpose()
        averager_regressor.fit(train_preds, y_train)
    return train_preds


def predict_multi_block(multi_block, attribute, test_df, test):
    y_pred = []
    for i in range(multi_block.number_blocks):
        embeddings = np.array(list(test_df[f"embeddings_{i}"])).reshape(len(test), -1)
        y_pred.append(multi_block.predict(i, attribute, embeddings))
    return y_pred


def score_multi_block(
    multi_block,
    attribute,
    test_df,
    test,
    y_test,
    y_pred,
    averager_regressor=None,
    average_function=np.mean,
    weights=None,
):
    if averager_regressor is not None:
        y_pred = np.array(y_pred).transpose()
        y_pred = averager_regressor.predict(y_pred)
    else:
        if weights is None:
            y_pred = average_function(y_pred, axis=0)
        y_pred = average_function(y_pred, weights)

    score = calculate_rmse_score_single(y_test, y_pred)
    return score


def objective(trial=None, splitter_n=2):
    # Window parameters
    use_sliding_window = False

    use_data_augmentation = False
    augmentation_csvs = [
        "gpt_neo_full_2022-10-27-20-36-14.csv",
    ]

    # block_size = trial.suggest_int("block_size", low=128, high=2048, step=128)
    block_size = 1152
    # step_size = trial.suggest_int("step_size", low=32, high=block_size, step=32)
    step_size = block_size // 2

    minimum_chunk_length = 10
    window_size = block_size

    # Only used if sliding window is not used
    # if splitter_n is None and trial is not None:
    #     splitter_n = trial.suggest_int("splitter_n", 1, 10)

    test_size = 0.2
    splits = 1

    def average_function(preds, weights):
        sum_ = 0.0
        denonimator = sum(weights)
        for idx, p in enumerate(preds):
            sum_ += weights[idx] * p
        return sum_ / denonimator

    class WeightingStrategy:
        def linear(n):
            return [i / n for i in range(n)]

        def lasso_cv(*args, **kwargs):
            return []

        def diminishing(n):
            return [1 / (i + 1) for i in range(n)]

        def diminishing_2(n):
            d = [1 / (i + 1) ** 2 for i in range(n)]
            d[0] = 0.5
            d[1] = 0.3
            d[2] = 0.2
            return d

        def step_decrease(n):
            return [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01][:n]

        def uniform(n):
            return [1 / n for _ in range(n)]

        def custom(*args):
            def c(_):
                l = []
                for i in args:
                    l.append(i)
                return l

            return c

        def linear_net(*args):
            return []

    def splitter(text):
        return split_text_into_n_parts(text, splitter_n, minimum_chunk_length)

    def splitter_window(text):
        return split_text_into_sliding_windows(
            text,
            window_size=window_size,
            step_size=step_size,
            minimum_chunk_length=minimum_chunk_length,
        )

    strategy_name = (
        f"splitter_window-{window_size}-{step_size}"
        if use_sliding_window
        else f"splitter-{splitter_n}"
    )
    if use_data_augmentation:
        strategy_name += (
            f"-augmentation-{augmentation_csvs[0]}-2-{len(augmentation_csvs)}"
        )
    if use_sliding_window:
        splitting_strategy = SplittingStrategy(
            splitter=splitter_window, name=strategy_name
        )
    else:
        splitting_strategy = SplittingStrategy(splitter=splitter, name=strategy_name)

    sentence_csv_dir = "./sentence_csvs"

    # Load the model
    full_df = pd.read_csv("/data/feedback-prize/train.csv")
    print("Original len(full_df)", len(full_df))

    if use_data_augmentation:
        for csv in augmentation_csvs:
            path = f"./generated_csvs/{csv}"
            full_df = pd.concat([full_df, pd.read_csv(path)], ignore_index=True)
        print("Augmented len(full_df)", len(full_df))

    print(full_df)

    model_info = ModelCatalog.DebertaV3
    multi_block_class = MultiBlockRidgeCV
    multi_block = multi_block_class(
        model=ModelStack(
            [
                model_info,
                # ModelCatalog.T5V1Large,
            ],
        ),
        number_blocks=splitter_n,
        labels=attributes,
    )

    for _ in range(2):
        try:
            smart_blockenizer(
                full_df,
                sentence_csv_dir,
                columns_mapping={
                    "text": "sentence_text",
                    "full_text": "full_text",
                    "id": "text_id",
                    "labels": attributes,
                },
                multi_head=multi_block,
                splitting_strategy=splitting_strategy,
            )
            break
        except KeyError:
            print("KeyError, retrying")
            pass

    if use_sliding_window:
        multi_block.set_number_blocks(max(full_df["number_blocks"]))
        print("Max number of blocks", multi_block.number_blocks)
    else:
        assert multi_block.number_blocks == max(full_df["number_blocks"])

    weighting_strategy = WeightingStrategy.uniform
    weights = weighting_strategy(multi_block.number_blocks)

    print("Weights >>>>", weights)

    print("Full DF POST Merge \n\n ------------------")
    print(full_df)

    best_scores = {}
    X = full_df["full_text"]

    for attribute in attributes:
        y = np.array(full_df[attribute])

        skf = StratifiedShuffleSplit(
            n_splits=splits, test_size=test_size, random_state=KFOLD_RANDOM_STATE
        )
        averager_regressor = None
        if weighting_strategy == WeightingStrategy.lasso_cv:
            averager_regressor = LassoCV()
        elif weighting_strategy == WeightingStrategy.linear_net:
            averager_regressor = LinearNet(multi_block.number_blocks)

        for train, test in skf.split(X, y):
            # Filter train DF
            train_df = full_df.filter(items=train, axis=0)
            y_train = np.array(train_df[attribute])

            # Filter test DF
            test_df = full_df.filter(items=test, axis=0)
            y_test = np.array(test_df[attribute])

            fit_multi_block(
                multi_block, attribute, train_df, train, y_train, averager_regressor
            )

            y_pred = predict_multi_block(multi_block, attribute, test_df, test)

            score = score_multi_block(
                multi_block,
                attribute,
                test_df,
                test,
                y_test,
                y_pred,
                averager_regressor,
                average_function,
                weights,
            )
            print(f"Score for {attribute} is {score}")

            if attribute not in best_scores:
                best_scores[attribute] = score
            else:
                best_scores[attribute] = min(score, best_scores[attribute])

    print("Best scores")
    print(best_scores)

    print("Average score")
    avg = np.mean(list(best_scores.values()))
    print(avg)
    return avg


use_optuna = False

if use_optuna:
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = (
        "sliding-window-block-1152-lassocv-deberta"  # Unique identifier of the study.
    )
    storage_name = "sqlite:///exploration_dbs/{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",  # we want to minimize the error :)
    )

    study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)
    print(study.best_trial)
else:
    objective(trial=None)
