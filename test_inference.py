import os
from datetime import datetime

import pandas as pd
from setfit import SetFitModel

from mongo_api import MongoDataAPIClient
from utils import (
    MCRMSECalculator,
    reverse_labels,
    round_border_score,
)

mongo_api = MongoDataAPIClient()

output_dir = "./small_sets/"
data_dir = "/data/feedback-prize/"
train_filepath = os.path.join(data_dir, "train.csv")
challenge_df_filepath = os.path.join(data_dir, "test.csv")
train_df = pd.read_csv(train_filepath)

attribute_experiments = {
    "cohesion": "cohesion_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "syntax": "syntax_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "phraseology": "phraseology_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "vocabulary": "vocabulary_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "grammar": "grammar_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "conventions": "conventions_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1"
}

models = {k: {} for k in attribute_experiments.keys()}
is_regression = True

print("Predicting full dataset...")
t0 = datetime.now()
for attribute, value in attribute_experiments.items():
    print("Processing attribute: ", attribute)
    model_path = f"/data/feedback-prize/models/{value}"
    is_regression = "LinearRegression" in model_path or "SGDRegressor" in model_path
    models[attribute]["is_regression"] = is_regression
    model = SetFitModel.from_pretrained(model_path)
    models[attribute]["model"] = model

    train_df[f"{attribute}_predictions"] = model.predict(train_df["full_text"].tolist())
    if is_regression:
        train_df[f"{attribute}_predictions"] = train_df[
            f"{attribute}_predictions"
        ].apply(lambda x: round_border_score(x))
    else:
        train_df[f"{attribute}_predictions"] = train_df[
            f"{attribute}_predictions"
        ].apply(lambda x: reverse_labels[x])


t1 = datetime.now()
print("Elapsed time to run inference on full dataset: ", t1 - t0)

train_df[
    [
        "text_id",
        "cohesion",
        "cohesion_predictions",
        "syntax",
        "syntax_predictions",
        "vocabulary",
        "vocabulary_predictions",
        "phraseology",
        "phraseology_predictions",
        "grammar",
        "grammar_predictions",
        "conventions",
        "conventions_predictions",
    ]
].to_csv("full_predictions.csv", index=False)
print(train_df.head())

full_preds_df = pd.read_csv("full_predictions.csv")
mcrmse_calculator = MCRMSECalculator()
mcrmse_calculator.compute_score_for_df(full_preds_df)
score = mcrmse_calculator.get_score()
print("MCRMSE score: ", score)
