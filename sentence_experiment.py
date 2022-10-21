import os
from first_iteration_setfit.utils import (
    calculate_rmse_score_attribute,
    split_df_into_sentences,
    split_text_into_n_parts,
    split_text_into_half,
    split_text_into_sliding_windows,
)
import pandas as pd
from datetime import datetime
from model_loader import load_model_with_dropout
from load_data import create_train_test_df
from utils import attributes, calculate_rmse_score

from sklearn.linear_model import LassoCV, RidgeCV, SGDRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from pre_trained_st_model import MultiHeadSentenceTransformerFactory
from model_catalog import ModelCatalog
from model_stacker import ModelStack


def unroll_sentence_df(
    sentence_df, embeddings, attribute, train_max_length=0, trained_model=None
):
    unrolled = []
    texts = {}
    max_length = 0
    for text_id, embedding, attribute_value in zip(
        sentence_df["text_id"], embeddings, sentence_df[attribute]
    ):
        if text_id not in texts:
            texts[text_id] = {
                "embeddings": [],
                "attributes": [],
            }
        texts[text_id]["embeddings"].extend(embedding)
        if trained_model:
            predicted_attribute = trained_model.predict(
                f"{attribute}_embeddings", [embedding]
            )[0]
            texts[text_id]["attributes"].append(predicted_attribute)
        else:
            texts[text_id]["attributes"].append(attribute_value)
        max_length = max(len(texts[text_id]["embeddings"]), max_length)

    if train_max_length > 0:
        safe_guard = train_max_length
    else:
        safe_guard = max_length * 4
    if safe_guard < max_length:
        raise ValueError(
            "Max length of test set is larger than train set, cannot fit model."
        )
    for text_id, text in texts.items():
        if len(text["embeddings"]) < safe_guard:
            text["embeddings"].extend([0] * (safe_guard - len(text["embeddings"])))

        unrolled.append(
            {
                "text_id": text_id,
                "embeddings": text["embeddings"],
                "attributes": text["attributes"],
                "features": []
                + text["embeddings"]
                + [sum(text["attributes"]) / len(text["attributes"])]
                + [max(text["attributes"])]
                + [min(text["attributes"])]
                + [len(text["attributes"])],
            }
        )

    unrolled_df = pd.DataFrame(unrolled)
    return unrolled_df, safe_guard


minimum_chunk_length = 10
window_size = 1024
step_size = window_size

splitter_n = 2  # Only used if sliding window is not used


class SplittingStrategy:
    def __init__(self, splitter, name):
        self.splitter = splitter
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def splitter(text):
    return split_text_into_n_parts(text, splitter_n, minimum_chunk_length)


def splitter_window(text):
    return split_text_into_sliding_windows(
        text,
        window_size=window_size,
        step_size=step_size,
        minimum_chunk_length=minimum_chunk_length,
    )


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
compare_full = True
# Load the model
train_df, test_df = create_train_test_df(test_size, "full")

model_info = ModelCatalog.DebertaV3
multi_head_class = MultiHeadSentenceTransformerFactory.create_class(
    RidgeCV,
)
multi_head = multi_head_class(
    model=ModelStack([model_info]),
)

sentence_train_df_path = (
    f"{sentence_csv_dir}/train_{test_size}_{splitting_strategy.name}.csv"
)
sentence_test_df_path = (
    f"{sentence_csv_dir}/test_{test_size}_{splitting_strategy.name}.csv"
)


try:
    sentence_train_df = pd.read_csv(sentence_train_df_path)
except Exception:
    sentence_train_df = split_df_into_sentences(train_df, splitting_strategy.splitter)
    sentence_train_df.to_csv(sentence_train_df_path, index=False)

try:
    sentence_test_df = pd.read_csv(sentence_test_df_path)
except Exception:
    sentence_test_df = split_df_into_sentences(test_df, splitting_strategy.splitter)
    sentence_test_df.to_csv(sentence_test_df_path, index=False)

if compare_full:
    text_label = "full_text"
    # print("Encoding full texts...")
    X_original_train_embeddings = multi_head.encode(
        list(train_df[text_label]),
        batch_size=32,
        type_path="train",
        use_cache=True,
    )
    X_original_test_embeddings = multi_head.encode(
        list(test_df[text_label]), batch_size=32, type_path="test", use_cache=True
    )
    original_preds_df = pd.DataFrame()
    for attribute in attributes:
        multi_head.fit(attribute, X_original_train_embeddings, train_df[attribute])
        preds = multi_head.predict(attribute, X_original_test_embeddings)
        original_preds_df[attribute] = preds
        original_score = calculate_rmse_score_attribute(
            attribute, test_df, original_preds_df
        )
        print(f"Original score ({attribute}):", original_score)
    all_original_score = calculate_rmse_score(
        test_df[attributes].values, original_preds_df[attributes].values
    )
    print("Overall original score:", all_original_score)


text_label = "sentence_text"
X_train = list(sentence_train_df[text_label])
X_test = list(sentence_test_df[text_label])

# print("Encoding sentences...")
X_train_embeddings = multi_head.encode(
    X_train,
    batch_size=32,
    type_path=f"train_splitter:{splitting_strategy.name}_test-size:{test_size}",
    use_cache=True,
)
X_test_embeddings = multi_head.encode(
    X_test,
    batch_size=32,
    type_path=f"test_splitter:{splitting_strategy.name}_test-size:{test_size}",
    use_cache=True,
)

preds_df = pd.DataFrame()
preds_df["text_id"] = test_df["text_id"]
preds_df["full_text"] = test_df["full_text"]

for attribute in attributes:
    y_train = sentence_train_df[attribute]
    y_test = sentence_test_df[attribute]

    unrolled_train_df, train_max_length = unroll_sentence_df(
        sentence_train_df, X_train_embeddings, attribute
    )

    X_train_features = unrolled_train_df["features"].tolist()

    print(len(X_train_embeddings), len(y_train), len(X_train_features), len(train_df))
    multi_head.fit(f"{attribute}_embeddings", X_train_embeddings, y_train)

    multi_head.fit(attribute, X_train_features, train_df[attribute])

    unrolled_test_df, _ = unroll_sentence_df(
        sentence_test_df,
        X_test_embeddings,
        attribute,
        train_max_length=train_max_length,
        trained_model=multi_head,
    )
    X_test_features = unrolled_test_df["features"].tolist()

    s = multi_head.score(attribute, X_test_features, test_df[attribute])

    preds = multi_head.predict(attribute, X_test_features)
    preds_df[attribute] = preds

    sentence_score = calculate_rmse_score_attribute(attribute, test_df, preds_df)

    print(f"RMSE Unrolled Sentences Embeddings Score ({attribute}):", sentence_score)

print(
    "Overall RMSE Unrolled Sentences Embeddings Score:",
    calculate_rmse_score(test_df[attributes].values, preds_df[attributes].values),
)
