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
    "cohesion": "cohesion_SGDRegressor_1_3024c1a4-e9a9-4633-b827-cf823ad33fbf_epoch_8",
    "syntax": "syntax_SGDRegressor_1_cff46e69-c9fb-4fb9-8e0f-317970261922_epoch_2",
    "vocabulary": "vocabulary_SGDRegressor_1_49accbc7-4ed2-44b4-a3cc-725ad9a134d2_epoch_5",
    "phraseology": "phraseology_SGDRegressor_1_2d0be900-d674-4b72-a5a1-e5278dd719d7_epoch_3",
    "grammar": "grammar_SGDRegressor_1_424fe4d1-7441-4ffe-9f2b-9b5a4107eca7_epoch_10",
    "conventions": "conventions_SGDRegressor_1_43eefaf6-9f6a-4d67-9e5b-5f1b4480dc62_epoch_11",
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
mcrmse_calculator.compute_columns(
    class_labels=[
        full_preds_df["cohesion"],
        full_preds_df["syntax"],
        full_preds_df["vocabulary"],
        full_preds_df["phraseology"],
        full_preds_df["grammar"],
        full_preds_df["conventions"],
    ],
    predictions=[
        full_preds_df["cohesion_predictions"],
        full_preds_df["syntax_predictions"],
        full_preds_df["vocabulary_predictions"],
        full_preds_df["phraseology_predictions"],
        full_preds_df["grammar_predictions"],
        full_preds_df["conventions_predictions"],
    ],
)
score = mcrmse_calculator.get_score()
print(score)
# mongo_api.register_score(
#     f"full_dataset_{str(list(attribute_experiments.values()))}", score
# )
