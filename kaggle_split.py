from first_iteration_setfit.utils import calculate_rmse_score_attribute
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


# Window parameters
minimum_chunk_length = 10
window_size = 512
step_size = 512

splitter_n = 2  # Only used if sliding window is not used

test_size = 0.1
splits = 1

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

sentence_df_path = f"{sentence_csv_dir}/train_full_{splitting_strategy.name}.csv"

try:
    sentence_df = pd.read_csv(sentence_df_path)
except Exception:
    sentence_df = split_df_into_sentences(full_df, splitting_strategy.splitter)
    sentence_df.to_csv(sentence_df_path, index=False)

X = np.array(full_df["full_text"])

text_label = "sentence_text"

X_sentences = np.array(sentence_df[text_label])

# print("Encoding sentences...")
X_embeddings = multi_head.encode(
    X_sentences,
    batch_size=32,
    type_path=f"full_splitter:{splitting_strategy.name}",
    use_cache=True,
)

unrolled_df, train_max_length = unroll_labelled_sentence_df_all(
    sentence_df, X_embeddings
)
print(unrolled_df.head())
X_unrolled_embeddings = np.array(unrolled_df["embeddings"])
for attribute in attributes:
    print(len(full_df), attribute)
    y = list(full_df[attribute])
    print("y: ", y[0:5])
    skf = StratifiedShuffleSplit(n_splits=splits, test_size=test_size)
    for train, test in skf.split(X, y):
        train_unrolled_df = unrolled_df.filter(train, axis=0)
        X_train_unrolled_embeddings = X_unrolled_embeddings[train]
        y_train = y[train]

        test_unrolled_df = unrolled_df.filter(test, axis=0)
        X_test_unrolled_embeddings = X_unrolled_embeddings[test]
        y_test = y[test]

        X_train_features = train_unrolled_df[f"{attribute}_features"]

        # print(X_test_unrolled_embeddings[0:5])s
        print(y_train[0:5])
        multi_head.fit(
            f"{attribute}_embeddings",
            X_train_unrolled_embeddings,
            y_train,
            # np.array(X_test_unrolled_embeddings),
            # np.array(y_test),
        )
        multi_head.fit(attribute, X_train_features, y_train)

        # Have to build the test features using the trained models
        X_test_features = infer_labels(
            test_unrolled_df, X_test_unrolled_embeddings, multi_head
        )

        y_pred = multi_head.predict(X_test_features[attribute])
        rmse = calculate_rmse_score_attribute(y_test, y_pred, attribute)
        print(f"{attribute} RMSE: {rmse}")

    # print(f"RMSE Unrolled Sentences Embeddings Score ({attribute}):", sentence_score)

print("Mean MCRMSE score: ", multi_head.get_mean_score())

# print(
#     "Overall RMSE Unrolled Sentences Embeddings Score:",
#     calculate_rmse_score(test_df[attributes].values, preds_df[attributes].values),
# )
