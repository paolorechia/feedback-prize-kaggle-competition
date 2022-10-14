import os
from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer, evaluation, losses, models
from sentence_transformers.evaluation import SimilarityFunction
from torch.utils.data import DataLoader

from load_data import create_attribute_stratified_split, sample_sentences_per_class
from model_catalog import Model
from sentence_pairing import (
    EvaluationDataset,
    TrainingDataset,
    create_continuous_sentence_pairs,
)


@dataclass
class TrainingContext:
    model: SentenceTransformer
    model_info: Model

    # Parameters
    warmup_steps: int
    weight_decay: float
    train_steps: int
    checkpoint_steps: int
    learning_rate: float
    num_epochs: int
    batch_size: int

    is_multitask: bool
    attributes: List[str]
    attribute: str

    unique_id: str
    text_label: str
    test_size: float
    input_dataset: str
    max_samples_per_class: int
    use_evaluator: bool

    checkout_dir: str
    output_dir: str


def auto_trainer(con: TrainingContext):
    if con.is_multitask:
        train_model_on_all_attributes(con)
    else:
        train_model_on_single_attribute(con)


def train_model_on_all_attributes(con: TrainingContext):
    output_path = os.path.join(
        con.output_dir,
        f"{con.model_info.model_name}-multitask-{str(con.unique_id[0:8])}",
    )

    train_objectives = []

    for attr in con.attributes:
        train_df, test_df = create_attribute_stratified_split(
            attr, con.test_size, dataset=con.input_dataset
        )

        small_subset = sample_sentences_per_class(
            train_df, attr, con.max_samples_per_class
        )

        training_dataset: TrainingDataset = create_continuous_sentence_pairs(
            small_subset,
            con.text_label,
            attr,
            con.model_info.model_truncate_length,
            "training",
        )
        # Define your train dataset, the dataloader and the train loss

        train_dataloader = DataLoader(
            training_dataset.training_pairs, shuffle=True, batch_size=con.batch_size
        )
        train_loss = losses.CosineSimilarityLoss(con.model)

        train_objectives.append((train_dataloader, train_loss))

    def evaluation_callback(score, epoch, steps):
        print(f"\n\n\tEpoch {epoch} - Evaluation score: {score} - Steps: {steps}\n\n")

    print("Starting training, results will be saved to: ", output_path)
    # Tune the model
    con.model.fit(
        train_objectives=train_objectives,
        epochs=con.num_epochs,
        # evaluator=evaluator # Not sure if this is supported in multi-task,
        warmup_steps=con.warmup_steps,
        weight_decay=con.weight_decay,
        output_path=output_path,
        save_best_model=True,
        steps_per_epoch=con.train_steps,
        optimizer_params={"lr": con.learning_rate},
        show_progress_bar=True,
        callback=evaluation_callback,
        checkpoint_save_steps=con.checkpoint_steps,
        checkpoint_path=os.path.join(
            con.checkout_dir,
            con.model_info.model_name,
            con.attribute,
            str(con.unique_id[0:8]),
        ),
    )


def train_model_on_single_attribute(con: TrainingContext):
    output_path = os.path.join(
        con.output_dir,
        f"{con.model_info.model_name}-{con.attribute}-{str(con.unique_id[0:8])}",
    )

    train_df, test_df = create_attribute_stratified_split(
        con.attribute, con.test_size, dataset=con.input_dataset
    )

    small_subset = sample_sentences_per_class(
        train_df, con.attribute, con.max_samples_per_class
    )

    print("Small subset size: ", len(small_subset))
    # Let's see how it looks like :)
    training_dataset: TrainingDataset = create_continuous_sentence_pairs(
        small_subset,
        con.text_label,
        con.attribute,
        con.model_info.model_truncate_length,
        "training",
    )
    training_dataset.print_sample(5)
    print("Training sentence pairs: ", len(training_dataset.training_pairs))

    if con.use_evaluator:
        eval_small_subset = sample_sentences_per_class(
            test_df, con.attribute, con.max_samples_per_class
        )

        evaluation_dataset: EvaluationDataset = create_continuous_sentence_pairs(
            eval_small_subset,
            con.text_label,
            con.attribute,
            con.model_info.model_truncate_length,
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
        training_dataset.training_pairs, shuffle=True, batch_size=con.batch_size
    )
    train_loss = losses.CosineSimilarityLoss(con.model)

    def evaluation_callback(score, epoch, steps):
        print(f"\n\n\tEpoch {epoch} - Evaluation score: {score} - Steps: {steps}\n\n")

    print("Starting training, results will be saved to: ", output_path)
    # Tune the model
    con.model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=con.num_epochs,
        evaluator=evaluator,
        warmup_steps=con.warmup_steps,
        weight_decay=con.weight_decay,
        output_path=output_path,
        save_best_model=True,
        steps_per_epoch=con.train_steps,
        optimizer_params={"lr": con.learning_rate},
        show_progress_bar=True,
        callback=evaluation_callback,
        checkpoint_save_steps=con.checkpoint_steps,
        checkpoint_path=os.path.join(
            con.checkout_dir,
            con.model_info.model_name,
            con.attribute,
            str(con.unique_id[0:8]),
        ),
    )
    print("Finished, results saved to: ", output_path)
