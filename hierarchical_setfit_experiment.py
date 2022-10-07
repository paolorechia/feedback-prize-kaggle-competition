from uuid import uuid4

import pandas as pd
from datasets import load_dataset
from setfit import SetFitModel
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor

from my_setfit_trainer import train
from utils import attributes

# # Use cohesion instead
# train_df = pd.read_csv("small_sets/cohesion_extremes.csv")

# train_df["cohesion_binary_label"] = train_df.apply(
#     lambda x: "average_or_below_average" if x.cohesion <= 3.0 else "above_average", axis=1
# )
# train_df.to_csv("small_sets/cohesion_extremes_binary.csv", index=False)


# test_df = pd.read_csv(f"small_sets/full_sampled_set.csv")
test_df  = pd.read_csv("/data/feedback-prize/train.csv")
test_df["cohesion_binary_label"] = test_df.apply(
    lambda x: "average_or_below_average" if x.cohesion <= 3.0 else "above_average", axis=1
)
attributes = ["cohesion"]

for attribute in attributes:
    print("Bootstraping setfit training for attribute: ", attribute)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"small_sets/{attribute}_binary.csv",
        },
    )

    head_model = LogisticRegression()
    is_regression = isinstance(head_model, LinearRegression) or isinstance(
        head_model, SGDRegressor
    )

    dataset["train"] = dataset["train"].rename_column("full_text", "text")

    if is_regression:
        dataset["train"] = dataset["train"].rename_column(attribute, "label")
    else:
        dataset["train"] = dataset["train"].rename_column(f"{attribute}_binary_label", "label")

    train_ds = dataset["train"]

    # Load SetFit model from Hub
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

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
    #>>> (0.95)**3
    #0.8573749999999999
    #>>> (0.95)**4
    #0.8145062499999999

    # Let's see what happens

    num_iters = 20
    num_epochs = 10
    batch_size = 16
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
        binary_labels=True,
    )

    for idx, epoch in enumerate(epoch_results):
        print(idx, epoch)
