"""Uses SentenceTransformer library directly instead."""

from subprocess import call
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import evaluation
from uuid import uuid4
from torch.utils.data import DataLoader
from load_data import create_attribute_stratified_split
from sentence_pairing import (
    create_continuous_sentence_pairs,
    TrainingDataset,
    EvaluationDataset,
)
from model_catalog import ModelCatalog

model_info = ModelCatalog.AllMiniLML6v2
# model_info = ModelCatalog.AllDistilrobertaV1

model_name = model_info.model_name
model_truncate_length = model_info.model_truncate_length
batch_size = model_info.recommended_batch_size

test_size = 0.9
text_label = "full_text"
input_dataset = "full"
attribute = "cohesion"
max_test_size = 500
train_steps = 1
use_evaluator = True
evaluator = None
unique_id = str(uuid4())
learning_rate = 2e-5
output_path = f"./st_output/{attribute}-{unique_id}"


# Define the model. Either from scratch or by loading a pre-trained model
model = SentenceTransformer(model_name)


train_df, test_df = create_attribute_stratified_split(
    attribute, 0.9, dataset=input_dataset
)

# Let's see how it looks like :)
training_dataset: TrainingDataset = create_continuous_sentence_pairs(
    train_df, text_label, attribute, model_truncate_length, "training"
)
training_dataset.print_sample(5)

if use_evaluator:
    evaluation_dataset: EvaluationDataset = create_continuous_sentence_pairs(
        test_df.sample(max_test_size),
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
    )

    evaluation_dataset.print_sample(3)

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(
    training_dataset.training_pairs, shuffle=True, batch_size=16
)
train_loss = losses.CosineSimilarityLoss(model)


def evaluation_callback(score, epoch, steps):
    print(f"Epoch {epoch} - Evaluation score: {score} - Steps: {steps}")


print("Starting training, results will be saved to: ", output_path)
# Tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    evaluator=evaluator,
    evaluation_steps=1000,
    warmup_steps=1000,
    output_path=output_path,
    save_best_model=True,
    steps_per_epoch=train_steps,
    optimizer_params={"lr": learning_rate},
    show_progress_bar=True,
    callback=evaluation_callback,
)
