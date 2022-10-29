import logging
import sys

from itertools import product
from xml.etree.ElementTree import QName
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
from folding import Fold, fit_fold, score_fold


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
    splits = 5

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

    X = full_df["full_text"]

    for attribute in attributes:
        if attribute != "cohesion":
            continue
        y = np.array(full_df[attribute])

        skf = StratifiedShuffleSplit(
            n_splits=splits, test_size=test_size, random_state=KFOLD_RANDOM_STATE
        )
        averager_regressor = None
        if weighting_strategy == WeightingStrategy.lasso_cv:
            averager_regressor = LassoCV()
        elif weighting_strategy == WeightingStrategy.linear_net:
            averager_regressor = LinearNet(multi_block.number_blocks)

        folds = []
        idx = 0
        # Creates the folds
        for train, test in skf.split(X, y):
            # Filter train DF
            train_df = full_df.filter(items=train, axis=0)
            y_train = np.array(train_df[attribute])
            train_text_ids = list(train_df["text_id"])

            # Filter test DF
            test_df = full_df.filter(items=test, axis=0)
            y_test = np.array(test_df[attribute])

            training_sequence = []
            for i in range(multi_block.number_blocks):
                training_sequence.append(y_train)

            y_trains = np.concatenate(training_sequence).reshape(
                multi_block.number_blocks, -1
            )

            new_fold = Fold(
                fold_number=idx,
                train_df=train_df,
                train_text_ids=train_text_ids,
                train_indices=train,
                y_train=y_train,
                y_trains=y_trains,
                y_test=y_test,
                test_df=test_df,
                test_indices=test,
            )
            folds.append(new_fold)
            idx += 1

        print("Folds ", folds)

        fine_tuned_labels = {}
        fp = f"fine_tuned_labels_kfold_experiment_{attribute}_{strategy_name}.csv"
        import os

        # # Create / load fine tuned labels
        # if os.path.exists(fp):
        #     fine_tuned_labels_df = pd.read_csv(fp)
        #     print("Loaded fine tuned labels df")
        #     for idx, row in fine_tuned_labels_df.iterrows():
        #         attr_labels = []
        #         for i in range(multi_block.number_blocks):
        #             attr_labels.append(row[f"{attribute}_{i}"])
        #         fine_tuned_labels[row["text_id"]] = np.array(attr_labels)
        # else:
        text_ids = list(full_df["text_id"])
        y_full = np.array(full_df[attribute])
        fine_tuned_labels_df = pd.DataFrame()
        fine_tuned_labels_df["full_text"] = full_df["full_text"]
        fine_tuned_labels_df["text_id"] = full_df["text_id"]
        fine_tuned_labels_df[f"{attribute}_original"] = full_df[attribute]
        sequence = []
        for i in range(multi_block.number_blocks):
            sequence.append(y_full)

        y_fulls = np.concatenate(sequence).reshape(multi_block.number_blocks, -1)
        for t in zip(text_ids, y_fulls):
            fine_tuned_labels[t[0]] = t[1]

        # Train the model
        for fold in folds:
            fine_tuned_labels[text_ids[i]] = y_fulls[:, i]

        equal_labels = 0
        different_labels = 0
        for _, item in fine_tuned_labels.items():
            if item[0] == item[1]:
                equal_labels += 1
            else:
                different_labels += 1
        print("Equal labels", equal_labels)
        print("Different labels", different_labels)

        combinations = []
        for i in range(multi_block.number_blocks):
            combinations.append(possible_labels)

        all_combinations = list(product(*combinations))

        combinations_to_try = []

        for labels in y_fulls.T:
            target_mean = sum(labels) / len(labels)
            valid_combinations = []
            for combination in all_combinations:
                if sum(combination) / len(combination) == target_mean:
                    valid_combinations.append(combination)
            combinations_to_try.append(valid_combinations)

        original_fold_scores = {}
        for fold in folds:
            fit_fold(fold, multi_block, attribute, averager_regressor)
            fold_score = score_fold(
                fold,
                multi_block,
                attribute,
                averager_regressor,
                average_function,
                weights,
            )
            original_fold_scores[fold.fold_number] = fold_score
            print(f"Fold {fold.fold_number} score: ", fold_score)
        mean_original_score = np.mean(list(original_fold_scores.values()))
        print("Mean original score", mean_original_score)

        best_mean_score = mean_original_score

        for idx, row in full_df.iterrows():
            # Debug mode :)
            if idx % 1000 != 0:
                continue
            print(idx)
            text_id = row["text_id"]

            current_labels = y_fulls[:, idx]
            combos = combinations_to_try[idx]
            print("Current labels", current_labels)
            for combo in combos:
                print("Testing combo", combo)
                scores = 0.0
                used_folds = 0
                for fold in folds:
                    if not fold.is_text_in_train_fold(text_id):
                        continue
                    used_folds += 1
                    fold.replace_label(text_id, list(combo))
                    fit_fold(fold, multi_block, attribute, averager_regressor)
                    fold_score = score_fold(
                        fold,
                        multi_block,
                        attribute,
                        averager_regressor,
                        average_function,
                        weights,
                    )
                    # print(f"Fold {fold.fold_number} score: {fold_score}")
                    scores += fold_score
                    fold.restore_labels()

                mean_score = scores / used_folds
                print("Mean score for combination: ", mean_score)
                sys.exit(1)
                if mean_score < best_mean_score:
                    print("Found better combination (than {})".format(best_mean_score))
                    best_mean_score = mean_score
                    y_fulls[:, idx] = combo
                    for fold in folds:
                        if not fold.is_text_in_train_fold(text_id):
                            continue
                        fold.replace_label(text_id, list(combo))
                        fold.save_labels()
                        fold.restore_labels()

        for i in range(multi_block.number_blocks):
            fine_tuned_labels_df[f"{attribute}_{i}"] = y_fulls[i]

        fine_tuned_labels_df.to_csv(fp)


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
