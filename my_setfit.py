from uuid import uuid4
import json

import pandas as pd
from datasets import load_dataset
from sentence_transformers.losses import (
    BatchAllTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    CosineSimilarityLoss,
)
from setfit import SetFitModel
from setfit.modeling import SupConLoss
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
from sklearn.model_selection import StratifiedShuffleSplit

from my_setfit_trainer import train
from utils import (
    attributes,
    labels,
    reverse_labels,
    split_df_into_sentences,
    break_sentences,
)
from mongo_api import Experiment

##################################################################################
########### Model/Training Config

minimum_chunk_length = 64
attention_probs_dropout_prob = 0.85
hidden_dropout_prob = 0.85
num_iters = 8
num_epochs = 5
learning_rate = 2e-5
unique_id = uuid4()
test_size = 0.8
attributes = ["cohesion"]
loss_function = CosineSimilarityLoss
use_sentences = False
batch_size = 512 if use_sentences else 64

model_name = "all-MiniLM-L6-v2"
setfit_model_max_length = 256

# model_ = "all-mpnet-base-v2"
# setfit_model_max_length = 384

# model_ = "all-distilroberta-v1"
# setfit_model_max_length = 512


model_ = model_name
# model_ = "/data/feedback-prize/models/cohesion_SGDRegressor_20_674b3f64-2841-402a-a0bd-5f0e5219ba0e_epoch_1"

with open(f"dropout_test/{model_}/config.json", "r") as f:
    model_config = json.load(f)

model_config["attention_probs_dropout_prob"] = attention_probs_dropout_prob
model_config["hidden_dropout_prob"] = hidden_dropout_prob

# Save the new config to the same file
with open(f"dropout_test/{model_}/config.json", "w") as f:
    json.dump(model_config, f, indent=4)

model = SetFitModel.from_pretrained(f"dropout_test/{model_}")


intermediary_csv_dir = "./intermediary_csvs"
split_csv_dirs = "./split_csvs"
train_df_path = f"{intermediary_csv_dir}/train_df.csv"
test_df_path = f"{intermediary_csv_dir}/test_df.csv"

model = SetFitModel.from_pretrained(model_)
head_model = RidgeCV()
is_regression = (
    isinstance(head_model, SGDRegressor)
    or isinstance(head_model, RidgeCV)
    or isinstance(head_model, LassoCV)
    or isinstance(head_model, ElasticNet)
    or isinstance(head_model, OrthogonalMatchingPursuit)
    or isinstance(head_model, BayesianRidge)
    or isinstance(head_model, AdaBoostRegressor)
    # print(train_df.head())
    # Assert that are no NaNs in the dataframe
)
experiment = Experiment(
    experiment_name="Nameless",
    base_model=model_name,
    head_model=head_model.__class__.__name__,
    unique_id=str(unique_id),
    num_iters=num_iters,
    epochs=num_epochs,
    learning_rate=learning_rate,
    test_size=test_size,
    attribute=attributes[0],
    use_binary=False,
    use_sentences=use_sentences,
    setfit_model_max_length=setfit_model_max_length,
    minimum_chunk_length=minimum_chunk_length,
    batch_size=batch_size,
    loss_function=loss_function.__name__,
    metric="MCRMSE By Sentence",
    train_score=0.0,
    test_score=0.0,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    hidden_dropout_prob=hidden_dropout_prob,
)


##################################################################################
########## Load data

# Training with small balanced sample set of data
# full_df_path = "./small_sets/full_sampled_set.csv"

# Training with full dataset
full_df_path = "/data/feedback-prize/train.csv"

intermediate_df_path = "/data/feedback-prize/intermediate.csv"
fold_df_path = "/data/feedback-prize/"
text_label = "full_text"

full_df = pd.read_csv(full_df_path)


##################################################################################
########### Train!
for attribute in attributes:

    text_label = "full_text"

    X = full_df[text_label]
    y = full_df[attribute]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=10)
    train_index, test_index = next(sss.split(X, y))

    train_df = full_df.filter(items=train_index, axis=0)
    test_df = full_df.filter(items=test_index, axis=0)
    split_train_df_path = f"{split_csv_dirs}/train_{attribute}_{test_size}.csv"
    split_test_df_path = f"{split_csv_dirs}/test_{attribute}_{test_size}.csv"

    train_df.to_csv(split_train_df_path, index=False)
    test_df.to_csv(split_test_df_path, index=False)

    if use_sentences:
        # Try to use cache to speedup bootstap a little bit :)
        train_df_path = f"{intermediary_csv_dir}/train_{attribute}_{test_size}_{setfit_model_max_length}_{minimum_chunk_length}.csv"
        test_df_path = f"{intermediary_csv_dir}/test{attribute}_{test_size}_{setfit_model_max_length}_{minimum_chunk_length}.csv"
        text_label = "sentence_text"
        try:
            train_df = pd.read_csv(train_df_path)
        except Exception:
            train_df = break_sentences(
                split_df_into_sentences(train_df),
                setfit_model_max_length,
                minimum_chunk_length,
            )
        try:
            test_df = pd.read_csv(test_df_path)
        except Exception:
            test_df = break_sentences(
                split_df_into_sentences(test_df),
                setfit_model_max_length,
                minimum_chunk_length,
            )

    rename_attr = attribute
    if not is_regression:
        rename_attr = f"{attribute}_label"
        train_df[attribute] = train_df[f"{attribute}_label"].apply(
            lambda x: reverse_labels[x]
        )
        test_df[attribute] = test_df[f"{attribute}_label"].apply(
            lambda x: reverse_labels[x]
        )

    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    print("Bootstraping setfit training for attribute: ", attribute)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_df_path,
        },
    )

    dataset["train"] = dataset["train"].rename_column(text_label, "text")
    dataset["train"] = dataset["train"].rename_column(rename_attr, "label")

    train_ds = dataset["train"]

    experiment_name = "{}_{}_{}_{}".format(
        attribute, head_model.__class__.__name__, num_iters, unique_id
    )
    experiment_name = "{}_model:{}_head:{}_id:{}".format(
        attribute,
        model_name,
        head_model.__class__.__name__,
        str(experiment.unique_id)[0:8],
    )
    experiment.experiment_name = experiment_name

    print("Running experiment {}".format(experiment_name))
    epoch_results = train(
        experiment=experiment,
        experiment_name=experiment_name,
        model=model,
        attribute=attribute,
        train_dataset=train_ds,
        train_dataframe=train_df,
        test_dataframe=test_df,
        loss_class=loss_function,
        num_iterations=num_iters,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        head_model=head_model,
        is_regression=is_regression,
        save_results=True,
    )

    for idx, epoch in enumerate(epoch_results):
        print(idx, epoch)
