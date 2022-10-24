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
print("len(full_df)", len(full_df))

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

X = np.array(full_df["full_text"])

best_scores = {}
for attribute in attributes:
    y = np.array(full_df[attribute])

    skf = StratifiedShuffleSplit(
        n_splits=splits, test_size=test_size, random_state=KFOLD_RANDOM_STATE
    )
    idx = 0
    for train, test in skf.split(X, y):
        # Filter train DF
        train_df = full_df.filter(items=train, axis=0)
        y_train = train_df[attribute]

        print(len(train_df), len(train))

        text_label = "sentence_text"

        train_sentence_df_path = f"{sentence_csv_dir}/train_{test_size}_fold_{splits}_{idx}_{attribute}_{splitting_strategy.name}.csv"

        try:
            train_sentence_df = pd.read_csv(train_sentence_df_path)
        except Exception:
            train_sentence_df = split_df_into_sentences(
                train_df, splitting_strategy.splitter
            )
            train_sentence_df.to_csv(train_sentence_df_path, index=False)

        X_train_sentences = np.array(train_sentence_df[text_label])

        # print("Encoding training sentences...")
        X_train_embeddings = multi_head.encode(
            X_train_sentences,
            batch_size=32,
            type_path=f"full_splitter_train_:{test_size}_{splitting_strategy.name}_attribute_{attribute}_splits_{splits}_{idx}",
            use_cache=True,
        )

        train_unrolled_df, train_max_length = unroll_labelled_sentence_df_all(
            train_sentence_df, X_train_embeddings
        )
        X_train_unrolled_embeddings = np.array(train_unrolled_df["embeddings"])

        # Train first head layer
        multi_head.fit(
            f"{attribute}_embeddings", X_train_embeddings, train_sentence_df[attribute]
        )

        X_train_features = train_unrolled_df[f"{attribute}_features"].tolist()
        # Train second head layer that uses the first layer
        multi_head.fit(attribute, X_train_features, y_train)

        # Now repeat everything for test split
        test_df = full_df.filter(items=test, axis=0)
        y_test = test_df[attribute]

        test_sentence_df_path = f"{sentence_csv_dir}/test_{test_size}_fold_{splits}_{idx}_{attribute}_{splitting_strategy.name}.csv"
        try:
            test_sentence_df = pd.read_csv(test_sentence_df_path)
        except Exception:
            test_sentence_df = split_df_into_sentences(
                full_df.filter(items=test, axis=0), splitting_strategy.splitter
            )
            test_sentence_df.to_csv(test_sentence_df_path, index=False)

        X_test_sentences = np.array(test_sentence_df[text_label])
        # print("Encoding test sentences...")
        X_test_embeddings = multi_head.encode(
            X_test_sentences,
            batch_size=32,
            type_path=f"full_splitter_test_:{test_size}-{splitting_strategy.name}_attribute_{attribute}_splits_{splits}_{idx}",
            use_cache=True,
        )

        test_unrolled_df = infer_labels(
            test_sentence_df, X_test_embeddings, multi_head, train_max_length, attribute
        )

        X_test_features = test_unrolled_df[f"{attribute}_features"].tolist()
        # print("X_test_features\n", X_test_features[0:5])
        # Predict
        y_preds = multi_head.predict(attribute, X_test_features)
        score = calculate_rmse_score_single(y_test, y_preds)
        print(f"RMSE {attribute}: {score}")
        if attribute not in best_scores:
            best_scores[attribute] = score
        else:
            if score < best_scores[attribute]:
                best_scores[attribute] = score
                print(f"A better score was found ({attribute}):", score)
        idx += 1

print("Heads Mean Embeddings MCRMSE score: ", multi_head.get_mean_score())

print("Best scores\n\n-------------------------------------\n\n")
mean = np.mean(list(best_scores.values()))
for key, item in best_scores.items():
    print(f"{key}: {item}")
print(f"Mean: {mean}")
