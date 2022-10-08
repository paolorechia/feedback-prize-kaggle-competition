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
    "cohesion": "cohesion_SGDRegressor_20_674b3f64-2841-402a-a0bd-5f0e5219ba0e_epoch_1",
    "syntax": "syntax_SGDRegressor_20_253724a6-e4b5-4c23-9b37-57632d492fae_epoch_1",
    "vocabulary": "vocabulary_SGDRegressor_20_a4790b8f-bd5a-466d-8a0b-52e0e7ff0532_epoch_1",
    "phraseology": "phraseology_SGDRegressor_20_799732cf-a9a8-47e6-81ed-af8eed2ef979_epoch_1",
    "grammar": "grammar_SGDRegressor_20_e6d11781-d6c6-4d11-b901-5a43bba39ff3_epoch_1",
    "conventions": "conventions_SGDRegressor_20_16f5a1ea-6255-43d7-8129-6b2241f1f3db_epoch_1",
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
