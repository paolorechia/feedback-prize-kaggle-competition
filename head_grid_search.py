import os

import pandas as pd
from setfit import SetFitModel
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    LassoCV,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    RidgeCV,
    SGDRegressor,
)

from my_setfit_trainer import evaluate
from utils import attributes, labels, reverse_labels


models_dir = "/data/feedback-prize/models"
pretrained_model = "cohesion_model:all-MiniLM-L6-v2_head:SGDRegressor_iters:20_batchSize:512_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d1a5_epoch_4"
model_ = os.path.join(models_dir, pretrained_model)
attribute = "cohesion"
head_models_to_try = [
    LogisticRegression(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    SGDRegressor(),
    RidgeCV(),
    LassoCV(),
    OrthogonalMatchingPursuit(),
    ElasticNet(),
    BayesianRidge(),
]


model = SetFitModel.from_pretrained(model_)
use_chunked_sentences = True
if use_chunked_sentences:
    fold_df_path = "/data/feedback-prize/sentence_fold"
else:
    fold_df_path = "/data/feedback-prize/"
train_path = os.path.join(fold_df_path, f"train_{attribute}.csv")
test_path = os.path.join(fold_df_path, f"test_{attribute}.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df[f"{attribute}_label"] = train_df.apply(
    lambda x: labels[str(x[attribute])], axis=1
)
test_df[f"{attribute}_label"] = test_df.apply(
    lambda x: labels[str(x[attribute])], axis=1
)

scores = []
X_train = train_df["full_text"]
embeddings = model.model_body.encode(X_train)

for head_model in head_models_to_try:
    is_regression = "Logistic" not in str(type(head_model))

    if is_regression:
        y_train = train_df[attribute]
    else:
        y_train = train_df[f"{attribute}_label"]

    model.model_head = head_model
    model.model_head.fit(embeddings, y_train)
    train_score = evaluate(model, is_regression, train_df, attribute)
    test_score = evaluate(model, is_regression, test_df, attribute)
    print(
        """
    Model: {}
    Train score: {}
    Test score: {}
    """.format(
            head_model, train_score, test_score
        )
    )

    scores.append((head_model, train_score, test_score))

for score in scores:
    print(score)
