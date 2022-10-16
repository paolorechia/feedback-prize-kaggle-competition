import os

import pandas as pd
from sentence_transformers import evaluation, losses
from sentence_transformers.evaluation import SimilarityFunction
from torch.utils.data import DataLoader

from experiment_schemas import TrainingContext
from load_data import create_attribute_stratified_split, sample_sentences_per_class
from mcrmse_evaluator import evaluate_mcrmse_multitask
from mongo_api import MongoDataAPIClient
from sentence_pairing import (
    EvaluationDataset,
    TrainingDataset,
    create_continuous_sentence_pairs,
)


def auto_trainer(con: TrainingContext):
    mongo_collection = "sentence_transformers"
    mongo_client = MongoDataAPIClient(mongo_collection)
    mongo_client.register_st_training_context(con)
    if con.is_multitask:
        train_model_on_all_attributes(con, mongo_client)
    else:
        train_model_on_single_attribute(con, mongo_client)


def train_model_on_all_attributes(
    con: TrainingContext, mongo_client: MongoDataAPIClient = None
):
    output_path = os.path.join(
        con.output_dir,
        f"{con.model_info.model_name}-multitask-{str(con.unique_id[0:8])}",
    )

    train_objectives = []

    all_attributes_eval_dataset = EvaluationDataset([], [], [])

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

        evaluation_dataset = create_evaluation_dataset_for_attribute(test_df, con, attr)
        all_attributes_eval_dataset = all_attributes_eval_dataset.merge(
            evaluation_dataset
        )

    evaluator = None
    if con.use_evaluator:
        evaluator = create_evaluator_from_evaluation_dataset(all_attributes_eval_dataset)

    def evaluation_callback(score, epoch, steps):
        print(f"\n\n\tEpoch {epoch} - Evaluation score: {score} - Steps: {steps}\n\n")

        mcrmse_scores = evaluate_mcrmse_multitask(
            dataset_text_attribute=con.text_label,
            test_size_from_experiment=con.test_size,
            input_dataset=con.input_dataset,
            st_model=con.model,
        )
        if mongo_client:
            mongo_client.append_training_context_scores(
                con.unique_id,
                evaluation_score=score,
                mcrmse_scores=mcrmse_scores,
            )

    print("Starting training, results will be saved to: ", output_path)
    # Tune the model
    con.model.fit(
        train_objectives=train_objectives,
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


def train_model_on_single_attribute(con: TrainingContext, mongo_client=None):
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

    evaluator = None
    if con.use_evaluator:
        evaluator_dataset = create_evaluation_dataset_for_attribute(test_df, con)
        evaluator = create_evaluator_from_evaluation_dataset(evaluator_dataset)

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


def create_evaluation_dataset_for_attribute(
    test_df: pd.DataFrame, con: TrainingContext, attr: str = ""
) -> EvaluationDataset:

    used_attr = attr if attr else con.attribute
    eval_small_subset = sample_sentences_per_class(
        test_df, used_attr, con.max_samples_per_class
    )

    evaluation_dataset: EvaluationDataset = create_continuous_sentence_pairs(
        eval_small_subset,
        con.text_label,
        used_attr,
        con.model_info.model_truncate_length,
        "evaluation",
    )
    return evaluation_dataset


def create_evaluator_from_evaluation_dataset(
    evaluation_dataset: EvaluationDataset,
) -> evaluation.EmbeddingSimilarityEvaluator:

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        evaluation_dataset.sentences1,
        evaluation_dataset.sentences2,
        evaluation_dataset.scores,
        show_progress_bar=True,
        name="evaluator_output_{model_name}",
        write_csv=True,
        main_similarity=SimilarityFunction.COSINE,
    )

    evaluation_dataset.print_sample(min(3, len(evaluation_dataset.scores)))
    print("Evaluation sentence pairs: ", len(evaluation_dataset.scores))
    return evaluator
