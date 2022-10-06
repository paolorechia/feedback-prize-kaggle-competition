import logging
import math
import sys
from typing import TYPE_CHECKING

import numpy as np
from datasets import load_dataset
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.losses.BatchHardTripletLoss import (
    BatchHardTripletLossDistanceFunction,
)
from setfit import SetFitModel
from setfit.modeling import SupConLoss, sentence_pairs_generation
from torch.utils.data import DataLoader

dataset = load_dataset(
    "csv",
    data_files={
        "train": "small_sets/cohesion.csv",
        "test": "small_sets/full_sampled_set.csv",
    },
)

dataset["train"] = dataset["train"].rename_column("cohesion_label", "label")
dataset["train"] = dataset["train"].rename_column("full_text", "text")

dataset["test"] = dataset["test"].rename_column("cohesion_label", "label")
dataset["test"] = dataset["test"].rename_column("full_text", "text")


train_ds = dataset["train"]
test_ds = dataset["test"]

print(train_ds)
print(test_ds)
# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")


if TYPE_CHECKING:
    from datasets import Dataset
    from setfit.modeling import SetFitModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add handler to sys stdout
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(
    model: SetFitModel = model,
    train_dataset: "Dataset" = train_ds,
    eval_dataset: "Dataset" = test_ds,
    loss_class=CosineSimilarityLoss,
    num_iterations: int = 20,
    num_epochs=10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    # sentence-transformers adaptation
    batch_size = batch_size
    x_train = train_dataset["text"]
    y_train = train_dataset["label"]
    train_examples = []

    for _ in range(num_iterations):
        train_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), train_examples
        )

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size
    )
    train_loss = loss_class(model.model_body)
    train_steps = len(train_dataloader)

    logger.info("Using loss class: {}".format(loss_class))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_examples)}")
    logger.info(f"  Num epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {train_steps}")
    logger.info(f"  Total train batch size = {batch_size}")

    warmup_steps = math.ceil(train_steps * 0.1)
    model.model_body.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        steps_per_epoch=train_steps,
        optimizer_params={"lr": learning_rate},
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    # Train the final classifier
    model.fit(x_train, y_train)


# Let's see what happens
train(num_iterations=1, num_epochs=1, batch_size=16, learning_rate=2e-5)
