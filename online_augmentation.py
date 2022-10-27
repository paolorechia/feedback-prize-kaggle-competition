import json
import logging
import sys
from random import random
from uuid import uuid4
from warnings import warn
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedShuffleSplit

from data_augmentation import (
    GPT2Generator,
    GPTNeoGenerator,
    generate_from_df,
    add_labels_to_df,
)
from model_catalog import ModelCatalog
from model_stacker import ModelStack
from my_nets import LinearNet
from pre_trained_st_model import MultiBlockRidgeCV
from seeds import KFOLD_RANDOM_STATE
from splitter import (
    SplittingStrategy,
    smart_blockenizer,
    split_text_into_n_parts,
    split_text_into_sliding_windows,
)
from utils import attributes, calculate_rmse_score_single

from torch.nn.functional import normalize
import torch


def loss_function(net_outputs, old_score, new_score):
    net_outputs = torch.tensor(net_outputs, dtype=torch.float32)
    mean = torch.mean(normalize(net_outputs), dtype=torch.float32)
    t = torch.div(new_score, old_score)
    t.requires_grad = True
    return t - mean


def objective(trial=None, splitter_n=1):
    # Window parameters
    use_sliding_window = False
    block_size = 1152
    step_size = block_size // 2

    minimum_chunk_length = 10
    window_size = block_size

    test_size = 0.2
    splits = 1

    # For now must be 1, because text generation might fail and lead to unaligned data
    generation_sample_size = 1

    import random

    # text_generation_seed = random.randint(0, 100000)
    # text_generator = GPTNeoGenerator(seed=text_generation_seed)
    text_generator = GPT2Generator()
    target_generated_datapoints = 10000
    max_generation_attempts = 20000
    generation_uuid = str(uuid4())
    minimum_improvement = 0.00001

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

    # Data augmentation script only supports one block
    assert multi_block.number_blocks == 1

    weighting_strategy = WeightingStrategy.lasso_cv
    weights = weighting_strategy(multi_block.number_blocks)

    print("Weights >>>>", weights)

    print("Full DF POST Merge \n\n ------------------")
    print(full_df)

    best_scores = {}
    X = full_df["full_text"]

    # ignore warnings
    warnings.filterwarnings("ignore")
    with open(f"generated_texts_{generation_uuid}.txt", "w") as f:
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

                # Simpler training loop in this file
                train_embeddings = np.array(list(train_df[f"embeddings_0"])).reshape(
                    len(train), -1
                )
                multi_block.fit(0, attribute, train_embeddings, y_train)

                test_embeddings = np.array(list(test_df[f"embeddings_0"])).reshape(
                    len(test), -1
                )
                y_pred = multi_block.predict(0, attribute, test_embeddings)

                score = calculate_rmse_score_single(y_test, y_pred)
                print(f"Score ({attribute}) prior generation is {score}")

                # Data Augmentation (Naive) Online Flow
                generated_datapoints = 0
                attempts = 0
                text_generator.model.train()
                optimizer = torch.optim.AdamW(
                    text_generator.model.parameters(), lr=1e-5
                )
                while (
                    generated_datapoints < target_generated_datapoints
                    and attempts < max_generation_attempts
                ):
                    random_sample = train_df.sample(n=generation_sample_size)
                    # print(random_sample)
                    add_labels_to_df(random_sample)
                    generated_df, net_outputs = generate_from_df(
                        random_sample, text_generator
                    )

                    # print(generated_df)
                    augmented_df = pd.concat([train_df.copy(), generated_df])
                    try:
                        new_y = np.array(generated_df[attribute])
                    except KeyError:
                        print("Failed to generate new data, retrying")
                        continue

                    # print(y_train.shape)
                    # print(new_y.shape)

                    augmented_y = np.append(y_train, new_y)

                    generated_embeddings = multi_block.encode(
                        generated_df["full_text"].to_list()
                    )
                    # print(generated_embeddings.shape)
                    # print(train_embeddings.shape)
                    augmented_X = np.vstack((train_embeddings, generated_embeddings))

                    multi_block.fit(0, attribute, augmented_X, augmented_y)

                    y_pred = multi_block.predict(0, attribute, test_embeddings)

                    new_score = calculate_rmse_score_single(y_test, y_pred)
                    print(f"Score ({attribute}) post generation is {new_score}")
                    attempts += 1
                    loss = loss_function(net_outputs[0], score, new_score)
                    optimizer.zero_grad()
                    print("Experimental Loss", loss)
                    loss.backward()
                    optimizer.step()

                    if new_score < (score - minimum_improvement):
                        score = new_score
                        print(
                            "Score improved in a significant way, accepting generated data"
                        )
                        train_df = augmented_df
                        y_train = augmented_y
                        train_embeddings = augmented_X

                        score = new_score
                        generated_datapoints += len(generated_df)
                        for _, row in generated_df.iterrows():
                            f.write(
                                json.dumps(
                                    {
                                        "full_text": row["full_text"],
                                        "cohesion": row["cohesion"],
                                        "syntax:": row["syntax"],
                                        "grammar": row["grammar"],
                                        "conventions": row["conventions"],
                                        "phraseology": row["phraseology"],
                                        "vocabulary": row["vocabulary"],
                                        "augmentation_used_on_attribute": attribute,
                                    },
                                )
                            )
                            f.write("\n")
                    # print("Score did not improve, discarding generated data")

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
