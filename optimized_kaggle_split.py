from splitter import (
    infer_labels,
    split_df_into_sentences,
    split_text_into_n_parts,
    split_text_into_sliding_windows,
    unroll_sentence_df,
    SplittingStrategy,
    unroll_labelled_sentence_df_all,
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

sentence_df_path = f"{sentence_csv_dir}/full_optimized_{splitting_strategy.name}.csv"

try:
    full_sentence_df = pd.read_csv(sentence_df_path)
except Exception:
    full_sentence_df = split_df_into_sentences(full_df, splitting_strategy.splitter)
    full_sentence_df.to_csv(sentence_df_path, index=False)


print("Length of full_sentence_df", len(full_sentence_df))
print(full_sentence_df)

embeddings = [
    np.array(e)
    for e in multi_head.encode(
        full_sentence_df["sentence_text"],
        batch_size=32,
        show_progress_bar=True,
        use_cache=f"full-dataframe-optimized-sentence-encoding-{splitting_strategy.name}",
    )
]
full_sentence_df["embeddings"] = embeddings
model_embedding_length = len(embeddings[0])

print("Model embedding length", model_embedding_length)

# Make each row in full sentence DF become a new column in the full df
subset = list(
    zip(
        full_sentence_df["text_id"],
        full_sentence_df["sentence_text"],
        full_sentence_df["embeddings"],
    )
)


blocks_dict = {}

num_blocks = 0
for tuple_ in subset:
    text_id = tuple_[0]
    sentence = tuple_[1]
    embeddings = tuple_[2]
    if text_id not in blocks_dict:
        blocks_dict[text_id] = []
        blocks_dict[text_id].append({"sentence": sentence, "embeddings": embeddings})
    else:
        blocks_dict[text_id].append({"sentence": sentence, "embeddings": embeddings})
    num_blocks = max(num_blocks, len(blocks_dict[text_id]))


block_columns = []
block_embeddings = []
n_blocks_column = []
for j in range(num_blocks):
    block_columns.append([])
    block_embeddings.append([])

required_length = num_blocks

for idx, row in full_df.iterrows():
    text_id = row["text_id"]
    blocks_ = blocks_dict[text_id]
    n_blocks_column.append(len(blocks_))
    # Pad with empty strings
    while len(blocks_) < required_length:
        blocks_.append({"sentence": "", "embeddings": np.zeros(model_embedding_length)})

    for j in range(num_blocks):
        block_columns[j].append(blocks_[j]["sentence"])
        block_embeddings[j].append(blocks_[j]["embeddings"])

full_df["number_blocks"] = n_blocks_column
for j in range(num_blocks):
    full_df[f"block_{j}"] = block_columns[j]
    full_df[f"embeddings_{j}"] = block_embeddings[j]

print("Full DF POST Merge \n\n ------------------")
print(full_df)
