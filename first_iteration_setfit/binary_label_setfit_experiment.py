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
import json
import sys

# A good accuracy to start with an hierarchical chain of models is 0.95
# Because by the end, your actual accuracy will drop to 0.81 and 0.85
# >>> (0.95)**3
# 0.8573749999999999
# >>> (0.95)**4
# 0.8145062499999999

attention_probs_dropout_prob = 0.5
hidden_dropout_prob = 0.5
num_iters = 20
num_epochs = 24
learning_rate = 2e-5
unique_id = uuid4()
test_size = 0.9
attributes = ["cohesion"]
use_sentences = True
batch_size = 512
loss_function = CosineSimilarityLoss
full_train_df_path = "/data/feedback-prize/train.csv"
# full_train_df_path = "./small_sets/full_sampled_set.csv"

# model_ = "all-mpnet-base-v2"
# setfit_model_max_length = 384

# model_ = "all-distilroberta-v1"
# setfit_model_max_length = 512


# To use a different model, first save the base config to a file
# model = SetFitModel.from_pretrained(f"sentence-transformers/{model_}")
# model.save_pretrained(f"dropout_test/{model_}")
# sys.exit(0)

model_ = "all-MiniLM-L6-v2"
setfit_model_max_length = 256

base_model = f"sentence-transformers/{model_}"
minimum_chunk_length = 64


with open(f"dropout_test/{model_}/config.json", "r") as f:
    model_config = json.load(f)

model_config["attention_probs_dropout_prob"] = attention_probs_dropout_prob
model_config["hidden_dropout_prob"] = hidden_dropout_prob

# Save the new config to the same file
with open(f"dropout_test/{model_}/config.json", "w") as f:
    json.dump(model_config, f, indent=4)

model = SetFitModel.from_pretrained(f"dropout_test/{model_}")


full_df = pd.read_csv(full_train_df_path)
binary_label = "cohesion_binary_label"
full_df[binary_label] = full_df.apply(
    lambda x: "average_or_below_average" if x.cohesion <= 3.0 else "above_average",
    axis=1,
)
intermediary_csv_dir = "./intermediary_csvs"
train_df_path = f"{intermediary_csv_dir}/train_df.csv"

experiment = Experiment(
    experiment_name="Binary cohesion label",
    base_model=base_model,
    head_model="LogisticRegression",
    unique_id=str(unique_id),
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
    loss_function=loss_function.__name__,
    metric="accuracy",
    train_score=0.0,
    test_score=0.0,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    hidden_dropout_prob=hidden_dropout_prob,
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
        train_df_path = f"{intermediary_csv_dir}/train_{attribute}_{test_size}_{setfit_model_max_length}_{minimum_chunk_length}.csv"
        test_df_path = f"{intermediary_csv_dir}/test{attribute}_{test_size}_{setfit_model_max_length}_{minimum_chunk_length}.csv"
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
        experiment=experiment,
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
