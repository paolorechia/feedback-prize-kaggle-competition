from splitter import (
    split_text_into_n_parts,
    split_text_into_sliding_windows,
    SplittingStrategy,
    smart_blockenizer,
)
import pandas as pd
from utils import attributes, calculate_rmse_score_single

from sklearn.model_selection import StratifiedShuffleSplit
from pre_trained_st_model import MultiBlockRidgeCV
from model_catalog import ModelCatalog
from model_stacker import ModelStack
import numpy as np
from seeds import KFOLD_RANDOM_STATE


def splitter(text):
    return split_text_into_n_parts(text, splitter_n, minimum_chunk_length)


def splitter_window(text):
    return split_text_into_sliding_windows(
        text,
        window_size=window_size,
        step_size=step_size,
        minimum_chunk_length=minimum_chunk_length,
    )


# Window parameters
minimum_chunk_length = 10
window_size = 512
step_size = 512
splitter_n = 2  # Only used if sliding window is not used

test_size = 0.1
splits = 10

use_sliding_window = False
if use_sliding_window:
    splitting_strategy = SplittingStrategy(
        splitter=splitter_window, name=f"splitter_window-{window_size}-{step_size}"
    )
else:
    splitting_strategy = SplittingStrategy(
        splitter=splitter, name=f"splitter-{splitter_n}"
    )

sentence_csv_dir = "./sentence_csvs"
compare_full = False

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
            ModelCatalog.T03B,
        ],
    ),
    number_blocks=splitter_n,
    labels=attributes,
)

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

        for i in range(splitter_n):
            embeddings = np.array(list(train_df[f"embeddings_{i}"])).reshape(
                len(train), -1
            )
            multi_block.fit(i, attribute, embeddings, y_train)

        # Predict
        y_pred = []
        for i in range(splitter_n):
            embeddings = np.array(list(test_df[f"embeddings_{i}"])).reshape(
                len(test), -1
            )
            y_pred.append(multi_block.predict(i, attribute, embeddings))

        y_pred = np.mean(y_pred, axis=0)
        score = calculate_rmse_score_single(y_test, y_pred)
        print(f"Score for {attribute} is {score}")

        if attribute not in best_scores:
            best_scores[attribute] = score
        else:
            best_scores[attribute] = min(score, best_scores[attribute])


print("Best scores")
print(best_scores)

print("Average score")
print(np.mean(list(best_scores.values())))
