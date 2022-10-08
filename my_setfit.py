from uuid import uuid4

import pandas as pd
from datasets import load_dataset
from setfit import SetFitModel
from setfit.modeling import SupConLoss
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sentence_transformers.losses import (
    BatchHardTripletLoss,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    BatchHardSoftMarginTripletLoss,
)

from my_setfit_trainer import train
from utils import attributes, labels

# test_df = pd.read_csv("small_sets/full_sampled_set.csv")
full_df_path = "/data/feedback-prize/train.csv"
intermediate_df_path = "/data/feedback-prize/intermediate.csv"

test_df = pd.read_csv(full_df_path)
test_df["cohesion_label"] = test_df.apply(lambda x: labels[str(x.cohesion)], axis=1)
test_df.to_csv(intermediate_df_path, index=False)

# Train on whole dataset (probably overfitting)
attributes = ["cohesion"]
for attribute in attributes:
    print("Bootstraping setfit training for attribute: ", attribute)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": intermediate_df_path,
        },
    )

    head_model = SGDRegressor()
    is_regression = isinstance(head_model, LinearRegression) or isinstance(
        head_model, SGDRegressor
    )

    dataset["train"] = dataset["train"].rename_column("full_text", "text")

    if is_regression:
        dataset["train"] = dataset["train"].rename_column(attribute, "label")
    else:
        dataset["train"] = dataset["train"].rename_column(f"{attribute}_label", "label")

    train_ds = dataset["train"]

    model_name = "all-MiniLM-L6-v2"
    # Load SetFit model from Hub
    model = SetFitModel.from_pretrained(f"sentence-transformers/{model_name}")

    # Let's see what happens

    num_iters = 20
    num_epochs = 10
    batch_size = 128
    learning_rate = 2e-5
    unique_id = uuid4()
    experiment_name = "{}_{}_{}_{}".format(
        attribute, head_model.__class__.__name__, num_iters, unique_id
    )

    print("Running experiment {}".format(experiment_name))
    epoch_results = train(
        experiment_name=experiment_name,
        model=model,
        train_dataset=train_ds,
        test_dataframe=test_df,
        num_iterations=num_iters,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        head_model=head_model,
        is_regression=is_regression,
        loss_class=BatchSemiHardTripletLoss,
    )

    for idx, epoch in enumerate(epoch_results):
        print(idx, epoch)
