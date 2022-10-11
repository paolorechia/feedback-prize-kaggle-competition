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
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.model_selection import StratifiedShuffleSplit

from my_setfit_trainer import train
from utils import attributes, split_df_into_sentences, break_sentences

from mongo_api import Experiment
# First level
# 1.0 1.5 2.0 2.5 3.0
# 3.5 4.0 4.5 5.0
# Second level (lower)
# 1.0 1.5 2.0
# 2.5 3.0
# Third level (upper-lower)
# 1.0
# 1.5 2.0
# Fourth level :(, e.g., for lower level needs 4 layers - How good should accuracy be?

# Second level (upper)
# 3.5 4.0
# 4.5 5.0
# Third level, e.g., for upper level needs 3 layers, how good should accuracy be?

# A good accuracy to start with an hierarchical chain of models is 0.95
# Because by the end, your actual accuracy will drop to 0.81 and 0.85
# >>> (0.95)**3
# 0.8573749999999999
# >>> (0.95)**4
# 0.8145062499999999

# Let's see what happens

# # Use cohesion instead
# train_df = pd.read_csv("small_sets/cohesion_extremes.csv")

# train_df["cohesion_binary_label"] = train_df.apply(
#     lambda x: "average_or_below_average" if x.cohesion <= 3.0 else "above_average", axis=1
# )
# train_df.to_csv("small_sets/cohesion_extremes_binary.csv", index=False)


# test_df = pd.read_csv(f"small_sets/full_sampled_set.csv")
# Load SetFit model from Hub
base_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SetFitModel.from_pretrained(base_model)
num_iters = 5
num_epochs = 5
learning_rate = 2e-5
unique_id = uuid4()
test_size = 0.9
attributes = ["cohesion"]
use_sentences = True
setfit_model_max_length = 256
minimum_chunk_length = 32
batch_size = 64
loss_function = CosineSimilarityLoss
full_df = pd.read_csv("/data/feedback-prize/train.csv")
binary_label = "cohesion_binary_label"
full_df[binary_label] = full_df.apply(
    lambda x: "average_or_below_average" if x.cohesion <= 3.0 else "above_average",
    axis=1,
)
intermediary_csv_dir = "./intermediary_csvs"
train_df_path = f"{intermediary_csv_dir}/train_df.csv"

experiment = Experiment(
    experiment_name="Binary cohesion label",
    base_model = base_model,
    head_model = "LogisticRegression",
    unique_id=unique_id,
    num_iters=num_iters,
    epochs=num_epochs,
    learning_rate=learning_rate,
    test_size=test_size,
    attribute=attributes[0],
    use_binary=True,
    use_sentences=use_sentences,
    setfit_model_max_length=setfit_model_max_length,
    minimum_chunk_length=minimum_chunk_length,
    batch_size=batch_size,
    loss_function=loss_function,
    metric="accuracy",
    train_score=0.0,
    test_score=0.0,
)
print(experiment)
for attribute in attributes:

    text_label = "full_text"

    X = full_df[text_label]
    y = full_df[attribute]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=10)
    train_index, test_index = next(sss.split(X, y))

    train_df = full_df.filter(items=train_index, axis=0)
    test_df = full_df.filter(items=test_index, axis=0)

    if use_sentences:
        # Try to use cache to speedup bootstap a little bit :)
        train_df_path = f"{intermediary_csv_dir}/train_{attribute}_{test_size}.csv"
        test_df_path = f"{intermediary_csv_dir}/test{attribute}_{test_size}.csv"
        text_label = "sentence_text"
        try:
            train_df = pd.read_csv(train_df_path)
        except Exception:
            train_df = break_sentences(
                split_df_into_sentences(train_df, binary_label=binary_label),
                setfit_model_max_length,
                minimum_chunk_length,
                binary_label=binary_label,
            )
        try:
            test_df = pd.read_csv(test_df_path)
        except Exception:
            test_df = break_sentences(
                split_df_into_sentences(test_df, binary_label=binary_label),
                setfit_model_max_length,
                minimum_chunk_length,
                binary_label=binary_label,
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

    head_model = LogisticRegression()
    dataset["train"] = dataset["train"].rename_column(text_label, "text")

    dataset["train"] = dataset["train"].rename_column(
        f"{attribute}_binary_label", "label"
    )
    train_ds = dataset["train"]

    experiment_name = "{}_{}_{}_{}".format(
        attribute, head_model.__class__.__name__, num_iters, unique_id
    )
    print("Running experiment {}".format(experiment_name))
    epoch_results = train(
        experiment=None,
        experiment_name=experiment_name,
        model=model,
        train_dataset=train_ds,
        test_dataframe=test_df,
        num_iterations=num_iters,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        head_model=head_model,
        is_regression=False,
        binary_labels=True,
        loss_class=loss_function,
        attribute=attribute,
        train_dataframe=train_df,
    )

    for idx, epoch in enumerate(epoch_results):
        print(idx, epoch)
