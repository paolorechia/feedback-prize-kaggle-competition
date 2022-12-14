import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from setfit import SetFitModel

from mongo_api import MongoDataAPIClient
from utils import MCRMSECalculator, reverse_labels, round_border_score

mongo_api = MongoDataAPIClient()

output_dir = "./small_sets/"
data_dir = "/data/feedback-prize/"
is_chunked_model = False
is_cohesion_model = False
sentences = {}
if is_chunked_model:
    print("Loading sentences from train.csv")
    chunked_train_filepath = os.path.join(data_dir, "sentence_chunked_train.csv")
    chunked_df = pd.read_csv(chunked_train_filepath)
    for index, row in tqdm(iterable=chunked_df.iterrows(), total=len(chunked_df)):
        text_id = row["text_id"]
        if text_id in sentences:
            sentences[text_id].append(row["sentence_text"])
        else:
            sentences[text_id] = [text_id]
    print("Loaded")

# Tets on full dataset
# train_filepath = os.path.join(data_dir, "train.csv")

# Test on fold
split_csv_dirs = "./split_csvs"
# Make sure to load file with the correct test :)
train_filepath = os.path.join(split_csv_dirs, "test_cohesion_0.8.csv")

train_df = pd.read_csv(train_filepath)


attribute_experiments = {
    "cohesion": "cohesion_model:all-distilroberta-v1_head:RidgeCV_id:ba8bb3bd_epoch_1",
    "syntax": "syntax_model:all-distilroberta-v1_head:RidgeCV_id:ba8bb3bd_epoch_1",
    "phraseology": "phraseology_model:all-distilroberta-v1_head:RidgeCV_id:ba8bb3bd_epoch_2",
    "vocabulary": "vocabulary_model:all-distilroberta-v1_head:RidgeCV_id:ba8bb3bd_epoch_1",
    "grammar": "grammar_model:all-distilroberta-v1_head:RidgeCV_id:ba8bb3bd_epoch_2",
    "conventions": "conventions_model:all-distilroberta-v1_head:RidgeCV_id:ba8bb3bd_epoch_2",
}

models = {k: {} for k in attribute_experiments.keys()}
is_regression = True

print("Predicting full dataset...")
t0 = datetime.now()
for attribute, value in tqdm(
    iterable=attribute_experiments.items(), total=len(attribute_experiments)
):
    print("Processing attribute: ", attribute)
    model_path = f"/data/feedback-prize/models/{value}"
    is_regression = "LogisticRegression" not in model_path
    models[attribute]["is_regression"] = is_regression
    model = SetFitModel.from_pretrained(model_path)
    models[attribute]["model"] = model

    # For now chunked model only supports regression models
    if is_chunked_model:
        print("Predicting chunked model for attribute: ", attribute)
        for index, row in tqdm(iterable=train_df.iterrows(), total=len(train_df)):
            text_id = row["text_id"]
            prediction = np.mean(
                [round_border_score(p) for p in model.predict(sentences[text_id])]
            )
            train_df.loc[index, f"{attribute}_predictions"] = prediction
    else:
        train_df[f"{attribute}_predictions"] = model.predict(
            train_df["full_text"].tolist()
        )
        if is_regression:
            train_df[f"{attribute}_predictions"] = train_df[
                f"{attribute}_predictions"
            ].apply(lambda x: round_border_score(x))
        else:
            train_df[f"{attribute}_predictions"] = train_df[
                f"{attribute}_predictions"
            ].apply(lambda x: reverse_labels[x])


t1 = datetime.now()
print("Elapsed time to run inference on dataset: ", t1 - t0)

if is_cohesion_model:
    train_df[["text_id", "cohesion", "cohesion_predictions"]].to_csv(
        "full_predictions.csv", index=False
    )
else:
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

print("Computing score...")
full_preds_df = pd.read_csv("full_predictions.csv")
mcrmse_calculator = MCRMSECalculator()
if is_cohesion_model:
    mcrmse_calculator.compute_column(
        full_preds_df["cohesion"], full_preds_df["cohesion_predictions"]
    )
else:
    mcrmse_calculator.compute_score_for_df(full_preds_df)
score = mcrmse_calculator.get_score()
print("MCRMSE score: ", score)
