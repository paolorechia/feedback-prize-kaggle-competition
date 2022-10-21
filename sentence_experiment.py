import os
from first_iteration_setfit.utils import (
    calculate_rmse_score_attribute,
    break_sentences,
    split_df_into_sentences,
    split_text_into_half,
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

test_size = 0.2
sentence_csv_dir = "./sentence_csvs"
attribute = "cohesion"
compare_full = False
# Load the model
train_df, test_df = create_train_test_df(test_size, "full")

model_info = ModelCatalog.DebertaV3
multi_head_class = MultiHeadSentenceTransformerFactory.create_class(
    RidgeCV,
    # max_iter=4000
)
multi_head = multi_head_class(
    model=ModelStack([model_info]),
)

if compare_full:
    text_label = "full_text"
    print("Encoding full texts...")
    X_original_train_embeddings = multi_head.encode(
        list(train_df[text_label]),
        batch_size=32,
        type_path="train",
        use_cache=True,
    )
    X_original_test_embeddings = multi_head.encode(
        list(test_df[text_label]), batch_size=32, type_path="test", use_cache=True
    )
    # print(X_original_test_embeddings[0])
    # print(X_original_test_embeddings.shape)
    multi_head.fit(attribute, X_original_train_embeddings, train_df[attribute])
    preds = multi_head.predict(attribute, X_original_test_embeddings)
    original_preds_df = pd.DataFrame()
    original_preds_df[attribute] = preds
    original_score = calculate_rmse_score_attribute(
        attribute, test_df, original_preds_df
    )

    print("Original score:", original_score)

print("Sentence breaking...")
mml = model_info.model_truncate_length
minimum_chunk_length = 64

sentence_train_df_path = (
    f"{sentence_csv_dir}/train_{attribute}_{test_size}_{mml}_{minimum_chunk_length}.csv"
)
sentence_test_df_path = (
    f"{sentence_csv_dir}/test{attribute}_{test_size}_{mml}_{minimum_chunk_length}.csv"
)
text_label = "sentence_text"
try:
    sentence_train_df = pd.read_csv(sentence_train_df_path)
except Exception:
    sentence_train_df = split_df_into_sentences(train_df, split_text_into_half)
    sentence_train_df.to_csv(sentence_train_df_path, index=False)

try:
    sentence_test_df = pd.read_csv(sentence_test_df_path)
except Exception:
    sentence_test_df = split_df_into_sentences(test_df, split_text_into_half)
    sentence_test_df.to_csv(sentence_test_df_path, index=False)

text_label = "sentence_text"
X_train = list(sentence_train_df[text_label])
y_train = sentence_train_df[attribute]
X_test = list(sentence_test_df[text_label])
y_test = sentence_test_df[attribute]


# print("Time to encode:", time_to_encode_in_seconds)

# print("Evaluating on attribute: ", attribute)
print("Encoding sentences...")
X_train_embeddings = multi_head.encode(
    X_train,
    batch_size=32,
    type_path=f"train-half-text-{test_size}",
    use_cache=True,
)
X_test_embeddings = multi_head.encode(
    X_test, batch_size=32, type_path=f"test-half-text-{test_size}", use_cache=True
)


def unroll_sentence_df(
    sentence_df, embeddings, attribute, train_max_length=0, trained_model=None
):
    unrolled = []
    texts = {}
    max_length = 0
    N = 768
    for text_id, embedding, attribute_value in zip(
        sentence_df["text_id"], embeddings, sentence_df[attribute]
    ):
        if text_id not in texts:
            texts[text_id] = {
                "embeddings": [],
                "attributes": [],
            }
        if len(texts[text_id]["embeddings"]) < N:
            texts[text_id]["embeddings"].extend(embedding)
            texts[text_id]["embeddings"] = texts[text_id]["embeddings"][:N]
        if trained_model:
            predicted_attribute = trained_model.predict(
                f"{attribute}_embeddings", [embedding]
            )[0]
            texts[text_id]["attributes"].append(predicted_attribute)
        else:
            texts[text_id]["attributes"].append(attribute_value)
        max_length = max(len(texts[text_id]["embeddings"]), max_length)

    print("Max length found: ", max_length)
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
                "features": [] + text["embeddings"]
                + [sum(text["attributes"]) / len(text["attributes"])]
                + [max(text["attributes"])]
                + [min(text["attributes"])]
                + [len(text["attributes"])],
            }
        )

    unrolled_df = pd.DataFrame(unrolled)
    return unrolled_df, safe_guard


unrolled_train_df, train_max_length = unroll_sentence_df(
    sentence_train_df, X_train_embeddings, attribute
)

print(unrolled_train_df.head())

X_train_features = unrolled_train_df["features"].tolist()

multi_head.fit(
    f"{attribute}_embeddings", X_train_embeddings, sentence_train_df[attribute]
)
multi_head.fit(attribute, X_train_features, train_df[attribute])

unrolled_test_df, _ = unroll_sentence_df(
    sentence_test_df,
    X_test_embeddings,
    attribute,
    train_max_length=train_max_length,
    trained_model=multi_head,
)
X_test_features = unrolled_test_df["features"].tolist()
print(unrolled_test_df.head())

s = multi_head.score(attribute, X_test_features, test_df[attribute])
print("Regressor Score:", s)

text_label = "full_text"
preds_df = pd.DataFrame()
preds_df["text_id"] = test_df["text_id"]
preds_df[text_label] = test_df[text_label]

print(preds_df.head())
preds = multi_head.predict(attribute, X_test_features)
preds_df[attribute] = preds

print(preds_df.head())

# preds_dict = {}
# for _, row in preds_df.iterrows():
#     if row["text_id"] not in preds_dict:
#         preds_dict[row["text_id"]] = []
#     preds_dict[row["text_id"]].append(row[attribute])

# preds_dict = {k: sum(v) / len(v) for k, v in preds_dict.items()}
# preds_dict = {k: max(v) for k, v in preds_dict.items()}
# preds_dict = {k: min(v) for k, v in preds_dict.items()}

# print(preds_df.head())

# new_preds_df = pd.DataFrame()
# new_preds_df["text_id"] = test_df["text_id"]

# # There is something wrong with this pipeline, it's missing values
# new_preds_df[attribute] = new_preds_df["text_id"].apply(lambda x: preds_dict[x])

# means_preds_df = means_preds_df.sort_values(by=["text_id"])
# original_test_df = original_test_df.sort_values(by=["text_id"])

# print(new_preds_df.head())
print(test_df.head())
sentence_score = calculate_rmse_score_attribute(attribute, test_df, preds_df)
# mean_score = calculate_rmse_score_attribute(attribute, test_df, new_preds_df)

print("RMSE Unrolled Sentences Embeddings Score:", sentence_score)
# print("RMSE Mean Score:", mean_score)
