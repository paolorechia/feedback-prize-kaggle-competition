import logging
import sys

from itertools import product
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import (
    MultiBlockRidgeCV,
    fit_multi_block,
    predict_multi_block,
    score_multi_block,
)
from seeds import KFOLD_RANDOM_STATE
from splitter import (
    SplittingStrategy,
    smart_blockenizer,
    split_text_into_n_parts,
    split_text_into_sliding_windows,
)
from utils import attributes, calculate_rmse_score_single, possible_labels
from sklearn.linear_model import LassoCV
from my_nets import LinearNet
from weight_strategy import WeightingStrategy, average_function


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
            fine_tuned_labels = pd.DataFrame()
            # Filter train DF
            train_df = full_df.filter(items=train, axis=0)
            y_train = np.array(train_df[attribute])

            # Filter test DF
            test_df = full_df.filter(items=test, axis=0)
            y_test = np.array(test_df[attribute])

            training_sequence = []
            for i in range(multi_block.number_blocks):
                training_sequence.append(y_train)

            y_trains = np.concatenate(training_sequence).reshape(
                multi_block.number_blocks, -1
            )

            combinations = []
            for i in range(multi_block.number_blocks):
                combinations.append(possible_labels)

            fine_tuned_labels["full_text"] = train_df["full_text"]
            fine_tuned_labels["text_id"] = train_df["text_id"]
            fine_tuned_labels[f"{attribute}_original"] = train_df[attribute]

            existing_combinations = list(product(*combinations))
            # print("Combinations to try ", existing_combinations)
            # print(len(train_df), y_train.shape)
            for idx, original_label in enumerate(y_train):
                print("idx", idx)
                fit_multi_block(
                    multi_block,
                    attribute,
                    train_df,
                    train,
                    y_train,
                    y_trains,
                    averager_regressor,
                )

                y_pred = predict_multi_block(multi_block, attribute, test_df, test)

                original_score = score_multi_block(
                    y_test,
                    y_pred,
                    averager_regressor,
                    average_function,
                    weights,
                )
                best_combination = None
                print(f"Current score for {attribute} is {original_score}")
                # print("Attempting to fine tune labels...")

                # TODO: generalize this to any number of blocks
                combinations_to_try = [
                    combination
                    for combination in existing_combinations
                    if (combination[0] + combination[1]) / 2 == y_train[idx]
                ]
                best_score = 100.0
                # print("Combinations to try ", combinations_to_try)
                for combination in combinations_to_try:
                    # print("Trying combination", combination)
                    # print(y_trains.shape)
                    # print(y_trains[:, idx].shape)
                    y_trains_copy = y_trains.copy()
                    y_trains_copy[:, idx] = np.array(combination)
                    fit_multi_block(
                        multi_block,
                        attribute,
                        train_df,
                        train,
                        y_train,
                        y_trains_copy,
                        averager_regressor,
                    )

                    y_pred = predict_multi_block(multi_block, attribute, test_df, test)

                    new_score = score_multi_block(
                        y_test,
                        y_pred,
                        averager_regressor,
                        average_function,
                        weights,
                    )
                    if new_score < best_score and new_score < original_score:
                        best_score = new_score
                        best_combination = combination
                        print("Found better score", new_score)
                if best_combination is not None:
                    y_trains[:, idx] = np.array(best_combination)
                    print("Original score", original_score)
                    print("Best score", best_score)
                    print("Best combination", best_combination)
            for i in range(multi_block.number_blocks):
                fine_tuned_labels[f"{attribute}_{i}"] = y_trains[i]
            fine_tuned_labels.to_csv(
                f"fine_tuned_labels_experiment_{attribute}_{strategy_name}.csv"
            )

            if attribute not in best_scores:
                best_scores[attribute] = original_score
            else:
                best_scores[attribute] = min(original_score, best_scores[attribute])

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
