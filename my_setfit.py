import pandas as pd
from datasets import load_dataset
from setfit import SetFitModel
from sklearn.linear_model import LinearRegression, LogisticRegression

from my_setfit_trainer import train

dataset = load_dataset(
    "csv",
    data_files={
        "train": "small_sets/cohesion.csv",
        "test": "small_sets/full_sampled_set.csv",
    },
)


head_model = LinearRegression()
is_regression = isinstance(head_model, LinearRegression)

dataset["train"] = dataset["train"].rename_column("full_text", "text")

if is_regression:
    dataset["train"] = dataset["train"].rename_column("cohesion", "label")
else:
    dataset["train"] = dataset["train"].rename_column("cohesion_label", "label")

train_ds = dataset["train"]
test_df = pd.read_csv("small_sets/full_sampled_set.csv")

# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Let's see what happens

num_iters = 20
experiment_name = "{}_{}".format(head_model.__class__.__name__, num_iters)

epoch_results = train(
    experiment_name=experiment_name,
    model=model,
    train_dataset=train_ds,
    test_dataframe=test_df,
    num_iterations=num_iters,
    num_epochs=2,
    batch_size=16,
    learning_rate=2e-5,
    head_model=head_model,
    is_regression=is_regression,
)

for idx, epoch in enumerate(epoch_results):
    print(idx, epoch)
