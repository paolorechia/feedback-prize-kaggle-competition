from splitter import (
    split_text_into_n_parts,
    split_text_into_sliding_windows,
    SplittingStrategy,
    smart_blockenizer,
)
import pandas as pd
from utils import attributes, calculate_rmse_score, calculate_rmse_score_single

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV
from pre_trained_st_model import MultiHeadSentenceTransformerFactory
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
splits = 5

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
multi_head_class = MultiHeadSentenceTransformerFactory.create_class(
    RidgeCV,
)
multi_head = multi_head_class(
    model=ModelStack(
        [
            model_info,
            # ModelCatalog.T03B,
        ],
    ),
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
    multi_head=multi_head,
    splitting_strategy=splitting_strategy,
)

print("Full DF POST Merge \n\n ------------------")
print(full_df)
