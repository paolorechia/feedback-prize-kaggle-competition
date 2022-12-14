import logging
import warnings
import math
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.losses.BatchHardTripletLoss import (
    BatchHardTripletLossDistanceFunction,
)
from setfit import SetFitModel
from setfit.modeling import SupConLoss, sentence_pairs_generation
from torch.utils.data import DataLoader

from utils import reverse_labels, round_border_score, calculate_rmse_score_attribute

if TYPE_CHECKING:
    from datasets import Dataset
    from setfit.modeling import SetFitModel

from mongo_api import MongoDataAPIClient, Experiment

MongoDataAPIClient()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add handler to sys stdout
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(
    experiment_name: str,
    attribute: str,
    model: SetFitModel,
    train_dataset: "Dataset",
    train_dataframe: DataFrame,
    test_dataframe: DataFrame,
    loss_class=CosineSimilarityLoss,
    experiment: Experiment = None,
    num_iterations: int = 20,
    num_epochs=10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    head_model=None,
    is_regression=False,
    binary_labels=False,
    save_results=True,
    use_sentences=False,
):
    # Init score tracker
    mongo_api = MongoDataAPIClient()

    # sentence-transformers adaptation
    batch_size = batch_size
    x_train = train_dataset["text"]
    y_train = train_dataset["label"]
    train_examples = []

    epoch_results = []
    # Overwrite default head model
    if head_model is not None:
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
    for current_epoch in range(1, num_epochs + 1):
        current_name = f"{experiment_name}_epoch_{current_epoch}"

        if current_epoch == 1:
            warmup_steps = warmup_steps
        else:
            warmup_steps = 0

        model.model_body.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=train_steps,
            optimizer_params={"lr": learning_rate},
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
        # Train the final classifier
        model.fit(x_train, y_train)

        # Evalute the model
        train_score = evaluate(
            model,
            is_regression,
            train_dataframe,
            attribute,
            binary_labels,
            is_sentences=use_sentences,
        )
        test_score = evaluate(
            model,
            is_regression,
            test_dataframe,
            attribute,
            binary_labels,
            is_sentences=use_sentences,
        )
        print(
            """
        Train score: {}
        Test score: {}
        """.format(
                train_score, test_score
            )
        )
        if save_results:
            model._save_pretrained(f"/data/feedback-prize/models/{current_name}")
        epoch_results.append((train_score, test_score))

        experiment.attribute = attribute
        experiment.train_score = train_score
        experiment.test_score = test_score
        experiment.epochs = current_epoch
        mongo_api.register_experiment(experiment=experiment)

    return epoch_results


def evaluate(
    model,
    is_regression,
    test_df,
    attribute,
    binary_labels=False,
    is_sentences=False,
):
    print("Evaluating on test dataset...")
    t0 = datetime.now()
    text_label = "sentence_text" if is_sentences else "full_text"

    if binary_labels:
        test_df[f"{attribute}_binary_prediction"] = model.predict(
            test_df[text_label].tolist()
        )
        accuracy = 0.0
        hits = 0
        errors = 0
        for index, row in test_df.iterrows():
            if (
                row[f"{attribute}_binary_prediction"]
                == row[f"{attribute}_binary_label"]
            ):
                hits += 1
            else:
                errors += 1

        accuracy = hits / (hits + errors)
        print("Accuracy: ", accuracy)
        return accuracy
    else:
        predictions_df = test_df.copy()
        predictions_df[attribute] = model.predict(test_df[text_label].tolist())
        if is_regression:
            predictions_df[attribute] = predictions_df[attribute].apply(
                lambda x: round_border_score(x)
            )
        else:
            predictions_df[attribute] = predictions_df[attribute].apply(
                lambda x: reverse_labels[x]
            )
        t1 = datetime.now()
        print(f"Time taken to predict: {t1 - t0}")

        score = calculate_rmse_score_attribute(attribute, test_df, predictions_df)
        print("MCRMSE score: ", score)
        return score
