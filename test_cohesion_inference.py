import os
from datetime import datetime

import pandas as pd
from setfit import SetFitModel

from mongo_api import MongoDataAPIClient
from utils import (
    MCRMSECalculator,
    fit_float_score_to_nearest_valid_point,
    reverse_labels,
)

mongo_api = MongoDataAPIClient()

output_dir = "./small_sets/"
data_dir = "/data/feedback-prize/"
train_filepath = os.path.join(data_dir, "train.csv")
challenge_df_filepath = os.path.join(data_dir, "test.csv")

experiment_name = "cohesion_SGDRegressor_4_850b88fc-6f21-469e-a6b1-99e22da0ce5f_epoch_2"
model_path = f"./models/cohesion/{experiment_name}"
is_regression = "LinearRegression" in model_path or "SGDRegressor" in model_path

model = SetFitModel.from_pretrained(model_path)

train_df = pd.read_csv(train_filepath)

print("Predicting full dataset...")
t0 = datetime.now()
train_df["cohesion_predictions"] = model.predict(train_df["full_text"].tolist())
if is_regression:
    train_df["cohesion_predictions"] = train_df["cohesion_predictions"].apply(
        lambda x: fit_float_score_to_nearest_valid_point(x)
    )
else:
    train_df["cohesion_predictions"] = train_df["cohesion_predictions"].apply(
        lambda x: reverse_labels[x]
    )


t1 = datetime.now()
print("Elapsed time to run inference on full dataset: ", t1 - t0)

train_df[["text_id", "cohesion", "cohesion_predictions"]].to_csv(
    "cohesion_predictions.csv", index=False
)
print(train_df.head())

cohesion_df = pd.read_csv("cohesion_predictions.csv")
mcrmse_calculator = MCRMSECalculator()
mcrmse_calculator.compute_column(
    cohesion_df["cohesion"], cohesion_df["cohesion_predictions"]
)
score = mcrmse_calculator.get_score()
print(score)
mongo_api.register_score("full_dataset_{}", score)
