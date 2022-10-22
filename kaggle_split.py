from first_iteration_setfit.utils import calculate_rmse_score_attribute
from splitter import (
    split_df_into_sentences,
    split_text_into_n_parts,
    split_text_into_sliding_windows,
    unroll_sentence_df,
    SplittingStrategy,
)
import pandas as pd
from utils import attributes, calculate_rmse_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV
from pre_trained_st_model import MultiHeadSentenceTransformerFactory
from model_catalog import ModelCatalog
from model_stacker import ModelStack
import numpy as np


def splitter(text):
    return split_text_into_n_parts(text, splitter_n, minimum_chunk_length)


def splitter_window(text):
    return split_text_into_sliding_windows(
        text,
        window_size=window_size,
        step_size=step_size,
        minimum_chunk_length=minimum_chunk_length,
    )


minimum_chunk_length = 10
window_size = 512
step_size = 512
splitter_n = 2  # Only used if sliding window is not used

use_sliding_window = False
if use_sliding_window:
    splitting_strategy = SplittingStrategy(
        splitter=splitter_window, name=f"splitter_window-{window_size}-{step_size}"
    )
else:
    splitting_strategy = SplittingStrategy(
        splitter=splitter, name=f"splitter-{splitter_n}"
    )

test_size = 0.2
sentence_csv_dir = "./sentence_csvs"
compare_full = False

# Load the model
full_df = pd.read_csv("/data/feedback-prize/train.csv")

model_info = ModelCatalog.DebertaV3
multi_head_class = MultiHeadSentenceTransformerFactory.create_class(
    RidgeCV,
)
multi_head = multi_head_class(
    model=ModelStack(
        [
            model_info,
            ModelCatalog.T03B,
        ],
    ),
)

sentence_df_path = f"{sentence_csv_dir}/train_full_{splitting_strategy.name}.csv"

try:
    sentence_df = pd.read_csv(sentence_df_path)
except Exception:
    sentence_df = split_df_into_sentences(full_df, splitting_strategy.splitter)
    sentence_df.to_csv(sentence_df_path, index=False)


text_label = "sentence_text"
X = np.array(list(sentence_df[text_label]))

# print("Encoding sentences...")
X_embeddings = multi_head.encode(
    X,
    batch_size=32,
    type_path=f"train_splitter:{splitting_strategy.name}_test-size:{test_size}",
    use_cache=True,
)

for attribute in attributes:
    y = sentence_df[attribute]
    test_size = 0.1
    splits = 1
    skf = StratifiedShuffleSplit(n_splits=splits, test_size=test_size)
    for train, test in skf.split(X, y):
        X_train = X[train]
        y_train = y[train]
        X_train_embeddings = X_embeddings[train]
        X_test_embeddings = X_embeddings[test]
        sentence_train_df_split = sentence_df.filter(items=train)
        train_df_split = full_df.filter(items=train)

        X_test = X[test]
        y_test = y[test]
        sentence_test_df_split = sentence_df.filter(items=test)
        test_df_split = full_df.filter(items=test)

        preds_df = pd.DataFrame()
        preds_df["text_id"] = test_df_split["text_id"]
        preds_df["full_text"] = test_df_split["full_text"]

        unrolled_train_df, train_max_length = unroll_sentence_df(
            sentence_train_df_split, X_test_embeddings, attribute
        )

        X_train_features = unrolled_train_df["features"].tolist()

        multi_head.fit_best_model(
            f"{attribute}_embeddings", X_train_embeddings, y_train
        )

        multi_head.fit(attribute, X_train_features, train_df_split[attribute])

        unrolled_test_df, _ = unroll_sentence_df(
            sentence_test_df_split,
            X_test_embeddings,
            attribute,
            train_max_length=train_max_length,
            trained_model=multi_head,
        )
        X_test_features = unrolled_test_df["features"].tolist()
        s = multi_head.score(attribute, X_test_features, test_df_split[attribute])

        preds = multi_head.predict(attribute, X_test_features)
        preds_df[attribute] = preds

        sentence_score = calculate_rmse_score_attribute(
            attribute, test_df_split, preds_df
        )

    print(f"RMSE Unrolled Sentences Embeddings Score ({attribute}):", sentence_score)

print("Mean MCRMSE score: ", multi_head.get_mean_score())

# print(
#     "Overall RMSE Unrolled Sentences Embeddings Score:",
#     calculate_rmse_score(test_df[attributes].values, preds_df[attributes].values),
# )
