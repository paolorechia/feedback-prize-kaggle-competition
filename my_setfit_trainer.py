import logging
import math
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel
from setfit.modeling import sentence_pairs_generation
from torch.utils.data import DataLoader

from utils import MCRMSECalculator, reverse_labels

if TYPE_CHECKING:
    from datasets import Dataset
    from setfit.modeling import SetFitModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add handler to sys stdout
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(
    experiment_name: str,
    model: SetFitModel,
    train_dataset: "Dataset",
    test_dataframe: DataFrame,
    loss_class=CosineSimilarityLoss,
    num_iterations: int = 20,
    num_epochs=10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    head_model=None,
    is_regression=False,
):
    # sentence-transformers adaptation
    batch_size = batch_size
    x_train = train_dataset["text"]
    y_train = train_dataset["label"]
    train_examples = []

    epoch_results = []
    # Overwrite default head model
    if head_model:
        model.model_head = head_model

    for _ in range(num_iterations):
        train_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), train_examples
        )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = loss_class(model.model_body)
    train_steps = len(train_dataloader)

    logger.info("Using loss class: {}".format(loss_class))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_examples)}")
    logger.info(f"  Num epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {train_steps}")
    logger.info(f"  Total train batch size = {batch_size}")

    warmup_steps = math.ceil(train_steps * 0.1)
    current_epoch = 1
    model.model_body.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        steps_per_epoch=train_steps,
        optimizer_params={"lr": learning_rate},
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    model._save_pretrained(f"./models/cohesion/{experiment_name}_epoch_{current_epoch}")

    # Train the final classifier
    model.fit(x_train, y_train)

    # Evalute the model
    score = evaluate(model, is_regression, test_dataframe)
    epoch_results.append(score)

    while current_epoch < num_epochs:
        current_epoch += 1
        model.model_body.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=train_steps,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True,
        )
        model._save_pretrained(
            f"./models/cohesion/{experiment_name}_epoch_{current_epoch}"
        )
        score = evaluate(model, is_regression, test_dataframe)
        epoch_results.append(score)
    return epoch_results


def evaluate(model, is_regression, test_df):
    print("Evaluating on test dataset...")
    t0 = datetime.now()
    test_df["cohesion_predictions"] = model.predict(test_df["full_text"].tolist())
    if not is_regression:
        test_df["cohesion_predictions"] = test_df["cohesion_predictions"].apply(
            lambda x: reverse_labels[x]
        )

    t1 = datetime.now()
    print(f"Time taken to predict: {t1 - t0}")

    mcrmse_calculator = MCRMSECalculator()
    mcrmse_calculator.compute_column(
        test_df["cohesion"], test_df["cohesion_predictions"]
    )
    score = mcrmse_calculator.get_score()
    print("MCRMSE score: ", score)
    return score
