"""Uses SentenceTransformer library directly instead."""

import pandas as pd
import os
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import evaluation
from sentence_transformers.evaluation import SimilarityFunction

from uuid import uuid4
from torch.utils.data import DataLoader
from load_data import create_attribute_stratified_split, sample_sentences_per_class
from sentence_pairing import (
    create_continuous_sentence_pairs,
    TrainingDataset,
    EvaluationDataset,
)
from model_catalog import ModelCatalog

model_info = ModelCatalog.AllMiniLML6v2
model_info = ModelCatalog.AllMpnetBasev1
model_info = ModelCatalog.AllDistilrobertaV1

model_name = model_info.model_name
model_truncate_length = model_info.model_truncate_length
batch_size = model_info.recommended_batch_size

text_label = "full_text"
input_dataset = "full"
test_dataset = "full"
attribute = "cohesion"

test_size = 0.5
max_samples_per_class = 8

use_evaluator = True
evaluator = None

unique_id = str(uuid4())

learning_rate = 2e-5
num_epochs = 6
checkpoint_steps = 50
weight_decay = 0.01
warmup_steps = 10
train_steps = 50

checkout_dir = "/data/feedback-prize/st-checkpoints/"
output_dir = "/data/feedback-prize/st-output/"
assert os.path.exists(checkout_dir)

output_path = os.path.join(
    output_dir, f"{model_name}-{attribute}-{str(unique_id[0:8])}"
)


# Define the model. Either from scratch or by loading a pre-trained model
model = SentenceTransformer(model_name)


train_df, test_df = create_attribute_stratified_split(
    attribute, test_size, dataset=input_dataset
)

small_subset = sample_sentences_per_class(train_df, attribute, max_samples_per_class)

print("Small subset size: ", len(small_subset))
# Let's see how it looks like :)
training_dataset: TrainingDataset = create_continuous_sentence_pairs(
    small_subset, text_label, attribute, model_truncate_length, "training"
)
training_dataset.print_sample(5)
print("Training sentence pairs: ", len(training_dataset.training_pairs))

if use_evaluator:
    eval_small_subset = sample_sentences_per_class(test_df, attribute, max_samples_per_class)

    evaluation_dataset: EvaluationDataset = create_continuous_sentence_pairs(
        eval_small_subset,
        text_label,
        attribute,
        model_truncate_length,
        "evaluation",
    )

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        evaluation_dataset.sentences1,
        evaluation_dataset.sentences2,
        evaluation_dataset.scores,
        show_progress_bar=True,
        name="evaluator_output_{model_name}",
        write_csv=True,
        main_similarity=SimilarityFunction.COSINE,
    )

    evaluation_dataset.print_sample(3)
    print("Evaluation sentence pairs: ", len(evaluation_dataset.scores))

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(
    training_dataset.training_pairs, shuffle=True, batch_size=batch_size
)
train_loss = losses.CosineSimilarityLoss(model)


def evaluation_callback(score, epoch, steps):
    print(f"\n\n\tEpoch {epoch} - Evaluation score: {score} - Steps: {steps}\n\n")


print("Starting training, results will be saved to: ", output_path)
# Tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    evaluator=evaluator,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    output_path=output_path,
    save_best_model=True,
    steps_per_epoch=train_steps,
    optimizer_params={"lr": learning_rate},
    show_progress_bar=True,
    callback=evaluation_callback,
    checkpoint_save_steps=checkpoint_steps,
    checkpoint_path=os.path.join(
        checkout_dir, model_name, attribute, str(unique_id[0:8])
    ),
)
print("Finished, results saved to: ", output_path)
