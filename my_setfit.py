from uuid import uuid4

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
from utils import attributes, labels, reverse_labels

##################################################################################
########### Model/Training Config

model_name = "all-MiniLM-L6-v2"
model_ = f"sentence-transformers/{model_name}"
# model_ = "/data/feedback-prize/models/cohesion_SGDRegressor_20_674b3f64-2841-402a-a0bd-5f0e5219ba0e_epoch_1"

model = SetFitModel.from_pretrained(model_)
head_model = SGDRegressor()
loss_function = CosineSimilarityLoss
num_iters = 20
num_epochs = 1
batch_size = 128
learning_rate = 2e-5
unique_id = uuid4()
attributes = ["cohesion"]
test_size = 0.2
is_regression = (
    isinstance(head_model, SGDRegressor)
    or isinstance(head_model, RidgeCV)
    or isinstance(head_model, LassoCV)
    or isinstance(head_model, ElasticNet)
    or isinstance(head_model, OrthogonalMatchingPursuit)
    or isinstance(head_model, BayesianRidge)
    or isinstance(head_model, AdaBoostRegressor)
    or isinstance(head_model, GradientBoostingRegressor)
    or isinstance(head_model, RandomForestRegressor)
)

##################################################################################
########## Load data
full_df_path = "/data/feedback-prize/train.csv"
intermediate_df_path = "/data/feedback-prize/intermediate.csv"
fold_df_path = "/data/feedback-prize/"

full_df = pd.read_csv(full_df_path)


##################################################################################
########### Train!
for attribute in attributes:
    X = full_df["full_text"]
    full_df[f"{attribute}_label"] = full_df.apply(
        lambda x: labels[str(x[attribute])], axis=1
    )
    if is_regression:
        y = full_df[attribute]

    else:
        y = full_df[f"{attribute}_label"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=10)
    train_index, test_index = next(sss.split(X, y))

    X_train = X.filter(items=train_index, axis=0)
    X_test = X.filter(items=test_index, axis=0)

    y_train = y.filter(items=train_index, axis=0)
    y_test = y.filter(items=test_index, axis=0)

    if is_regression:
        train_df = pd.DataFrame({"full_text": X_train, attribute: y_train})
        test_df = pd.DataFrame({"full_text": X_test, attribute: y_test})

    else:
        train_df = pd.DataFrame(
            {"full_text": X_train, f"{attribute}_label": y_train, attribute: y_train}
        )
        train_df[attribute] = train_df[f"{attribute}_label"].apply(
            lambda x: reverse_labels[x]
        )
        test_df = pd.DataFrame(
            {"full_text": X_test, f"{attribute}_label": y_test, attribute: y_test}
        )
        test_df[attribute] = test_df[f"{attribute}_label"].apply(
            lambda x: reverse_labels[x]
        )
    train_df.to_csv(intermediate_df_path, index=False)

    train_df.to_csv(fold_df_path + f"train_{attribute}.csv", index=False)
    test_df.to_csv(fold_df_path + f"test_{attribute}.csv", index=False)

    print("Bootstraping setfit training for attribute: ", attribute)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": intermediate_df_path,
        },
    )

    dataset["train"] = dataset["train"].rename_column("full_text", "text")

    if is_regression:
        dataset["train"] = dataset["train"].rename_column(attribute, "label")
    else:
        dataset["train"] = dataset["train"].rename_column(f"{attribute}_label", "label")

    train_ds = dataset["train"]

    experiment_name = (
        "{}_head:{}_iters:{}_batchSize:{}_lossFunction:{}_testSize:{}_id:{}".format(
            attribute,
            head_model.__class__.__name__,
            num_iters,
            batch_size,
            loss_function.__name__,
            test_size,
            str(unique_id)[0:4],
        )
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
        save_results=True,
    )

    for idx, epoch in enumerate(epoch_results):
        print(idx, epoch)
