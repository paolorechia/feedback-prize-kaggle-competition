import os

import pandas as pd
from sentence_transformers import evaluation, losses
from sentence_transformers.evaluation import SimilarityFunction
from torch.utils.data import DataLoader

from cuda_mem_report import report_cuda_memory
from experiment_schemas import TrainingContext
from load_data import (
    create_attribute_stratified_split,
    create_train_test_df,
    sample_sentences_per_class,
)
from mcrmse_evaluator import (
    evaluate_mcrmse_multitask_optimized,
    evaluate_mcrmse_single_attribute,
)
from mongo_api import MongoDataAPIClient
from sentence_pairing import (
    EvaluationDataset,
    TrainingDataset,
    create_continuous_sentence_pairs,
)


def auto_trainer(con: TrainingContext):
    mongo_collection = con.mongo_collection
    mongo_client = None
    if con.save_results_to_mongo:
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

    train_df, test_df = create_train_test_df(con.test_size, con.input_dataset)

    for attr in con.attributes:
        small_subset = sample_sentences_per_class(
            train_df, attr, con.max_samples_per_class
        )

        training_dataset: TrainingDataset = create_continuous_sentence_pairs(
            small_subset,
            con.text_label,
            attr,
            con.model_info.model_truncate_length,
            "training",
            distance_calculator=con.distance_function,
        )
        # Define your train dataset, the dataloader and the train loss

        train_dataloader = DataLoader(
            training_dataset.training_pairs,
            shuffle=True,
            batch_size=con.training_batch_size,
        )
        train_loss = con.loss_function_class(con.model)

        train_objectives.append((train_dataloader, train_loss))

        if con.use_evaluator and not con.skip_correlation_metric:
            evaluation_dataset = create_evaluation_dataset_for_attribute(
                test_df, con, attr
            )
            all_attributes_eval_dataset = all_attributes_eval_dataset.merge(
                evaluation_dataset
            )

    evaluator = None
    if con.use_evaluator:
        if con.skip_correlation_metric:
            # Create dummy evaluator, so we can use the evaluator callback
            evaluator = create_dummy_evaluator()
        else:
            evaluator = create_evaluator_from_evaluation_dataset(
                all_attributes_eval_dataset,
                batch_size=con.evaluation_batch_size,
            )

    def evaluation_callback(score, epoch, _):
        print(f"\n\n\tEpoch {epoch}\n\n")
        if not con.skip_correlation_metric:
            print(f"\t\tEvaluation score: {score}\n\n")

        if con.evaluate_mcmse:
            mcrmse_scores = evaluate_mcrmse_multitask_optimized(
                train_df=train_df,
                test_df=test_df,
                dataset_text_attribute=con.text_label,
                input_train_dataset=con.input_dataset,
                test_size_from_experiment=con.test_size,
                st_model=con.model,
                encoding_batch_size=con.evaluation_batch_size,
            )
            info = report_cuda_memory(verbose=False)
            if mongo_client:
                mongo_client.append_training_context_scores(
                    con.unique_id,
                    evaluation_score=-1,
                    mcrmse_scores=mcrmse_scores,
                    memory_usage=info.used,
                )

    # print("Starting training, results will be saved to: ", output_path)
    # Tune the model
    con.model.fit(
        train_objectives=train_objectives,
        epochs=con.num_epochs,
        evaluator=evaluator,
        warmup_steps=con.warmup_steps,
        weight_decay=con.weight_decay,
        output_path=output_path,
        save_best_model=False,
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
    # Try to delete everything that could possibly use GPU memory
    del evaluator
    del all_attributes_eval_dataset
    del train_dataloader
    del train_loss
    del training_dataset
    del train_objectives


def train_model_on_single_attribute(
    con: TrainingContext, mongo_client: MongoDataAPIClient = None
):
    output_path = os.path.join(
        con.output_dir,
        f"{con.model_info.model_name}-{con.attribute}-{str(con.unique_id[0:8])}",
    )
    all_attributes_eval_dataset = EvaluationDataset([], [], [])

    train_df, test_df = create_train_test_df(con.test_size, con.input_dataset)
    small_subset = sample_sentences_per_class(
        train_df, con.attribute, con.max_samples_per_class
    )
    train_objectives = []
    training_dataset: TrainingDataset = create_continuous_sentence_pairs(
        small_subset,
        con.text_label,
        con.attribute,
        con.model_info.model_truncate_length,
        "training",
        distance_calculator=con.distance_function,
    )
    # Define your train dataset, the dataloader and the train loss

    train_dataloader = DataLoader(
        training_dataset.training_pairs,
        shuffle=True,
        batch_size=con.training_batch_size,
    )
    train_loss = con.loss_function_class(con.model)

    train_objectives.append((train_dataloader, train_loss))

    if con.use_evaluator and not con.skip_correlation_metric:
        evaluation_dataset = create_evaluation_dataset_for_attribute(
            test_df, con, con.attribute
        )
        all_attributes_eval_dataset = all_attributes_eval_dataset.merge(
            evaluation_dataset
        )

    evaluator = None
    if con.use_evaluator:
        if con.skip_correlation_metric:
            # Create dummy evaluator, so we can use the evaluator callback
            evaluator = create_dummy_evaluator()
        else:
            evaluator = create_evaluator_from_evaluation_dataset(
                all_attributes_eval_dataset,
                batch_size=con.evaluation_batch_size,
            )

    def evaluation_callback(score, epoch, _):
        print(f"\n\n\tEpoch {epoch}\n\n")
        if not con.skip_correlation_metric:
            print(f"\t\tEvaluation score: {score}\n\n")

        if con.evaluate_mcmse:
            mcrmse_score = evaluate_mcrmse_single_attribute(
                train_df=train_df,
                test_df=test_df,
                dataset_text_attribute=con.text_label,
                attribute=con.attribute,
                st_model=con.model,
                encoding_batch_size=con.evaluation_batch_size,
            )
            print(mcrmse_score)
            info = report_cuda_memory(verbose=False)
            if mongo_client:
                mongo_client.append_training_context_attribute_score(
                    con.unique_id,
                    attribute=con.attribute,
                    attribute_score=mcrmse_score,
                    memory_usage=info.used,
                )

    con.model.fit(
        train_objectives=train_objectives,
        epochs=con.num_epochs,
        evaluator=evaluator,
        warmup_steps=con.warmup_steps,
        weight_decay=con.weight_decay,
        output_path=output_path,
        save_best_model=False,
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
        distance_calculator=con.distance_function,
    )
    return evaluation_dataset


def create_evaluator_from_evaluation_dataset(
    evaluation_dataset: EvaluationDataset, batch_size: int = 16
) -> evaluation.EmbeddingSimilarityEvaluator:

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        evaluation_dataset.sentences1,
        evaluation_dataset.sentences2,
        evaluation_dataset.scores,
        show_progress_bar=True,
        name="evaluator_output_{model_name}",
        write_csv=False,
        main_similarity=SimilarityFunction.COSINE,
        batch_size=batch_size,
    )

    # evaluation_dataset.print_sample(min(3, len(evaluation_dataset.scores)))
    # print("Evaluation sentence pairs: ", len(evaluation_dataset.scores))
    return evaluator


def create_dummy_evaluator() -> evaluation.EmbeddingSimilarityEvaluator:
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        ["a"],
        ["b"],
        ["0.5"],
        show_progress_bar=False,
        write_csv=False,
        main_similarity=SimilarityFunction.COSINE,
        batch_size=1,
    )
    return evaluator
