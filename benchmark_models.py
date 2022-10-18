import json

import pandas as pd

from load_data import create_train_test_df
from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import MultiHeadSentenceTransformerModelRidgeCV
from experiment_schemas import ModelBenchmark
from utils import attributes, calculate_rmse_score
from datetime import datetime

train_df, test_df = create_train_test_df(test_size=0.2, dataset="full")

X_train = list(train_df["full_text"])
X_test = list(test_df["full_text"])


model_scores = []
for model in [
    ModelCatalog.T03B
]:
    stack = ModelStack([model])

    multi_head = MultiHeadSentenceTransformerModelRidgeCV(stack)

    t0 = datetime.now()
    X_train_embeddings = multi_head.encode(X_train, batch_size=32)
    print(X_train_embeddings.shape)
    X_test_embeddings = multi_head.encode(X_test, batch_size=32)
    print(X_test_embeddings.shape)

    t1 = datetime.now()
    time_to_encode_in_seconds = (t1 - t0).total_seconds()

    print("Time to encode:", time_to_encode_in_seconds)
    preds_df = pd.DataFrame()
    preds_df["text_id"] = test_df["text_id"]
    preds_df["full_text"] = test_df["full_text"]

    for attribute in attributes:
        print("Evaluating on attribute: ", attribute)
        multi_head.fit(attribute, X_train_embeddings, train_df[attribute])
        s = multi_head.score(attribute, X_test_embeddings, test_df[attribute])
        print("Regressor Score:", s)
        preds = multi_head.predict(attribute, X_test_embeddings)
        preds_df[attribute] = preds

    score = calculate_rmse_score(
        test_df[attributes].values, preds_df[attributes].values
    )
    print("RMSE Score:", score)
    model_scores.append(
        ModelBenchmark(
            rmse_score=score,
            model_name=stack.stack[0].info.model_name,
            time_to_encode_in_seconds=time_to_encode_in_seconds,
        ).__dict__()
    )

with open("model_scores.json", "w") as f:
    json.dump(model_scores, f)
