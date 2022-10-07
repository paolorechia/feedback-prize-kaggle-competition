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
    "cohesion": "cohesion_LinearRegression_10_87dc1fd1-0e96-4317-8aa7-2bf2cf29a29f_epoch_20",
    "syntax": "syntax_LinearRegression_10_5d6ce138-ea24-41f2-b652-61c099b13070_epoch_1",
    "vocabulary": "vocabulary_LinearRegression_10_8c00aa80-fc17-4be2-83be-5b043b88b109_epoch_14",
    "phraseology": "phraseology_LinearRegression_10_016fdaf3-f0a3-4010-b5ac-2085337f120b_epoch_20",
    "grammar": "grammar_LinearRegression_10_1bf9a006-5dd3-46fc-b7d1-686cd20265fc_epoch_19",
    "conventions": "conventions_LinearRegression_10_20460b6c-bb36-408c-9276-08c8257d2ff2_epoch_7",
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
