from uuid import uuid4

import pandas as pd
from datasets import load_dataset
from setfit import SetFitModel
from setfit.modeling import SupConLoss
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sentence_transformers.losses import (
    BatchHardTripletLoss,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    BatchHardSoftMarginTripletLoss,
    CosineSimilarityLoss,
)

from my_setfit_trainer import train
from utils import attributes, labels

##################################################################################
########### Model/Training Config

model_name = "all-MiniLM-L6-v2"
model_ = f"sentence-transformers/{model_name}"
# model_ = "/data/feedback-prize/models/cohesion_SGDRegressor_20_674b3f64-2841-402a-a0bd-5f0e5219ba0e_epoch_1"

model = SetFitModel.from_pretrained(model_)
head_model = SGDRegressor()
loss_function = CosineSimilarityLoss
num_iters = 10
num_epochs = 5
batch_size = 128
learning_rate = 2e-5
unique_id = uuid4()
attributes = ["cohesion"]
test_size = 0.5

##################################################################################
########## Load/Prepare datasets
full_df_path = "/data/feedback-prize/train.csv"
intermediate_df_path = "/data/feedback-prize/intermediate.csv"

full_df = pd.read_csv(full_df_path)
full_df["cohesion_label"] = full_df.apply(lambda x: labels[str(x.cohesion)], axis=1)
full_df.to_csv(intermediate_df_path, index=False)
X = full_df["full_text"]
y = full_df["cohesion"]

sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=10)
train_index, test_index = next(sss.split(X, y))

X_train = X.filter(items=train_index, axis=0)
X_test = X.filter(items=test_index, axis=0)
y_train = y.filter(items=train_index, axis=0)
y_test = y.filter(items=test_index, axis=0)

train_df = pd.DataFrame({"full_text": X_train, "cohesion": y_train})
test_df = pd.DataFrame({"full_text": X_test, "cohesion": y_test})

train_df.to_csv(intermediate_df_path, index=False)


##################################################################################
########### Train!
for attribute in attributes:
    print("Bootstraping setfit training for attribute: ", attribute)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": intermediate_df_path,
        },
    )

    is_regression = isinstance(head_model, LinearRegression) or isinstance(
        head_model, SGDRegressor
    )

    dataset["train"] = dataset["train"].rename_column("full_text", "text")

    if is_regression:
        dataset["train"] = dataset["train"].rename_column(attribute, "label")
    else:
        dataset["train"] = dataset["train"].rename_column(f"{attribute}_label", "label")

    train_ds = dataset["train"]

    experiment_name = "{}_{}_{}_{}_{}_{}".format(
        attribute,
        head_model.__class__.__name__,
        num_iters,
        unique_id,
        loss_function.__name__,
        test_size,
    )

    print("Running experiment {}".format(experiment_name))
    epoch_results = train(
        experiment_name=experiment_name,
        model=model,
        attribute=attribute,
        train_dataset=train_ds,
        train_dataframe=train_df,
        test_dataframe=test_df,
        num_iterations=num_iters,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        head_model=head_model,
        is_regression=is_regression,
        loss_class=loss_function,
    )

    for idx, epoch in enumerate(epoch_results):
        print(idx, epoch)
