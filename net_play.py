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
from my_nets import ConvolutionalNet, LinearNet


def objective(trial=None, splitter_n=2):
    use_sliding_window = False

    block_size = 1152
    step_size = block_size // 2

    minimum_chunk_length = 10
    window_size = block_size

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

    if use_sliding_window:
        splitting_strategy = SplittingStrategy(
            splitter=splitter_window, name=f"splitter_window-{window_size}-{step_size}"
        )
    else:
        splitting_strategy = SplittingStrategy(
            splitter=splitter, name=f"splitter-{splitter_n}"
        )

    sentence_csv_dir = "./sentence_csvs"

    # Load the model
    full_df = pd.read_csv("/data/feedback-prize/train.csv")
    print("Original len(full_df)", len(full_df))
    print(full_df)

    model_info = ModelCatalog.DebertaV3
    multi_block_class = MultiBlockRidgeCV
    multi_block = multi_block_class(
        model=ModelStack(
            [
                model_info,
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

    print("Full DF POST Merge \n\n ------------------")
    print(full_df)

    best_scores = {}
    X = full_df["full_text"]

    for attribute in attributes:
        y = np.array(full_df[attribute])

        skf = StratifiedShuffleSplit(
            n_splits=splits, test_size=test_size, random_state=KFOLD_RANDOM_STATE
        )

        for train, test in skf.split(X, y):
            # Filter train DF
            train_df = full_df.filter(items=train, axis=0)
            y_train = np.array(train_df[attribute])

            # Filter test DF
            test_df = full_df.filter(items=test, axis=0)
            y_test = np.array(test_df[attribute])

            train_embeddings_matrix = []
            for _, row in train_df.iterrows():

                embeddings = []
                for i in range(multi_block.number_blocks):
                    embeddings.extend(row[f"embeddings_{i}"])
                embeddings = np.array(embeddings).reshape(-1)
                train_embeddings_matrix.append(embeddings)

            train_embeddings_matrix = np.array(train_embeddings_matrix)
            print("train_embeddings_matrix.shape", train_embeddings_matrix.shape)
            train_embeddings_matrix.reshape(len(train), -1)
            print("train_embeddings_matrix.shape", train_embeddings_matrix.shape)

            test_embeddings_matrix = []
            for _, row in test_df.iterrows():

                embeddings = []
                for i in range(multi_block.number_blocks):
                    embeddings.extend(row[f"embeddings_{i}"])
                embeddings = np.array(embeddings).reshape(-1)
                test_embeddings_matrix.append(embeddings)

            test_embeddings_matrix = np.array(test_embeddings_matrix)
            print("test_embeddings_matrix.shape", test_embeddings_matrix.shape)
            test_embeddings_matrix.reshape(len(test), -1)
            print("test_embeddings_matrix.shape", test_embeddings_matrix.shape)

            # net = LinearNet(
            #     len(train_embeddings_matrix[0]), hidden_size=2048, dropout=0.0
            # )
            net = ConvolutionalNet(
                len(train_embeddings_matrix[0]), num_channels=128, dropout=0.5
            )

            net.train_with_eval(
                X=train_embeddings_matrix,
                Y=y_train,
                X_eval=test_embeddings_matrix,
                Y_eval=y_test,
                batch_size=32,
                epochs=100,
                lr=0.001,
            )

            y_pred = net.predict(test_embeddings_matrix)
            score = calculate_rmse_score_single(y_test, y_pred)
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
