from warnings import warn

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from load_data import create_attribute_stratified_split
from pre_trained_st_model import MultiHeadSentenceTransformerModelRidgeCV
from utils import MCRMSECalculator, attributes, round_border_score


def evaluate_mcrmse_multitask(
    dataset_text_attribute: str,
    test_size_from_experiment: float,
    input_dataset: str,
    st_model: SentenceTransformer,
    evaluation_batch_size: int = 32,
    debug: bool = False,
):
    warn("This function is deprecated. Use evaluate_mcrmse_multitask_optimized instead")
    if not isinstance(st_model, SentenceTransformer):
        raise ValueError("Expected SentenceTransformer")

    st_model_ridge_cv = MultiHeadSentenceTransformerModelRidgeCV(st_model)

    predictions_df = pd.DataFrame()

    scores = {}

    if debug:
        attribs = [attributes[0]]
    else:
        attribs = attributes

    print("Evaluating MCRMSE on multitask...")
    for attribute in attribs:
        print("Evaluating on attribute: ", attribute)
        train_df, test_df = create_attribute_stratified_split(
            attribute, test_size_from_experiment, dataset=input_dataset
        )
        X_train = list(train_df[dataset_text_attribute])
        y_train = list(train_df[attribute])

        X_test = list(test_df[dataset_text_attribute])
        y_test = list(test_df[attribute])

        if predictions_df.columns.empty:
            predictions_df["text_id"] = test_df["text_id"]
            predictions_df[dataset_text_attribute] = test_df[dataset_text_attribute]

        predictions_df[f"{attribute}_predictions"] = test_df[attribute]

        X_train_embeddings = st_model.encode(X_train, batch_size=evaluation_batch_size)
        X_test_embeddings = st_model.encode(X_test, batch_size=evaluation_batch_size)

        st_model_ridge_cv.fit(attribute, X_train_embeddings, y_train)
        score = st_model_ridge_cv.score(attribute, X_test_embeddings, y_test)
        print("Score:", score)

        predictions = st_model_ridge_cv.predict(attribute, X_test_embeddings)

        predictions = [round_border_score(p) for p in predictions]

        print(f"Prediction samples ({attribute}) (prediction / label):")
        for j in range(5):
            print(predictions[j], y_test[j])
        mcrmse_calculator = MCRMSECalculator()
        mcrmse_calculator.compute_column(y_test, predictions)
        score = mcrmse_calculator.get_score()
        print(f"MCRMSE ({attribute}):", score)
        scores[attribute] = score
        predictions_df[attribute] = predictions
        # Clear embeddings from memory
        del X_train_embeddings
        del X_test_embeddings
        torch.cuda.empty_cache()

    if not debug:
        # Compute MCRMSE for all attributes
        # Merge each predictions_df with the previous one
        # deduplicating the text_id column and averaging the scores
        # for the same text_id
        predictions_df = predictions_df.groupby("text_id").mean().reset_index()
        mcrmse_calculator = MCRMSECalculator()
        mcrmse_calculator.compute_score_for_df(predictions_df)
        score = mcrmse_calculator.get_score()
        scores["all"] = score
        print("MCRMSE (all attributes):", score)
    return scores


def evaluate_mcrmse_multitask_optimized(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_text_attribute: str,
    input_train_dataset: str,
    test_size_from_experiment: float,
    st_model: SentenceTransformer,
    encoding_batch_size: int = 32,
):
    if not isinstance(st_model, SentenceTransformer):
        raise ValueError("Expected SentenceTransformer")

    st_model_ridge_cv = MultiHeadSentenceTransformerModelRidgeCV(st_model)

    predictions_df = pd.DataFrame()

    scores = {}

    X_train = list(train_df[dataset_text_attribute])
    X_test = list(test_df[dataset_text_attribute])

    # Optimized encodes only once
    print("Encoding test dataset...")
    X_test_embeddings = st_model.encode(
        X_test, batch_size=encoding_batch_size, show_progress_bar=True
    )
    print("Encoding train dataset...")
    X_train_embeddings = st_model.encode(
        X_train, batch_size=encoding_batch_size, show_progress_bar=True
    )

    print("Evaluating MCRMSE on multitask...")
    for attribute in attributes:
        print("Evaluating on attribute: ", attribute)
        y_train = list(train_df[attribute])
        y_test = list(test_df[attribute])

        st_model_ridge_cv.fit(attribute, X_train_embeddings, y_train)
        score = st_model_ridge_cv.score(attribute, X_test_embeddings, y_test)
        print("Score:", score)

        predictions = st_model_ridge_cv.predict(attribute, X_test_embeddings)
        predictions = [round_border_score(p) for p in predictions]

        print(f"Prediction samples ({attribute}) (prediction / label):")
        for j in range(5):
            print(predictions[j], y_test[j])

        if predictions_df.columns.empty:
            predictions_df["text_id"] = test_df["text_id"]
            predictions_df[dataset_text_attribute] = test_df[dataset_text_attribute]

        predictions_df[attribute] = test_df[attribute]
        predictions_df[f"{attribute}_predictions"] = predictions

        mcrmse_calculator = MCRMSECalculator()
        mcrmse_calculator.compute_column(y_test, predictions)
        score = mcrmse_calculator.get_score()
        print(f"MCRMSE ({attribute}):", score)
        scores[attribute] = score

        # Clear embeddings from memory
        torch.cuda.empty_cache()

    del X_test_embeddings
    del X_train_embeddings

    predictions_df = predictions_df.groupby("text_id").mean().reset_index()
    mcrmse_calculator = MCRMSECalculator()
    mcrmse_calculator.compute_score_for_df(predictions_df)
    score = mcrmse_calculator.get_score()
    scores["all"] = score
    print("MCRMSE (all attributes):", score)
    print(scores)
    return scores


def evaluate_mcrmse_single_attribute(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_text_attribute: str,
    attribute: str,
    st_model: SentenceTransformer,
    encoding_batch_size: int = 32,
) -> float:
    if not isinstance(st_model, SentenceTransformer):
        raise ValueError("Expected SentenceTransformer")

    st_model_ridge_cv = MultiHeadSentenceTransformerModelRidgeCV(st_model)

    predictions_df = pd.DataFrame()

    X_train = list(train_df[dataset_text_attribute])
    X_test = list(test_df[dataset_text_attribute])

    # Optimized encodes only once
    print("Encoding test dataset...")
    X_test_embeddings = st_model.encode(
        X_test, batch_size=encoding_batch_size, show_progress_bar=True
    )
    print("Encoding train dataset...")
    X_train_embeddings = st_model.encode(
        X_train, batch_size=encoding_batch_size, show_progress_bar=True
    )

    print("Evaluating MCRMSE on multitask...")
    print("Evaluating on attribute: ", attribute)
    y_train = list(train_df[attribute])
    y_test = list(test_df[attribute])

    st_model_ridge_cv.fit(attribute, X_train_embeddings, y_train)
    score = st_model_ridge_cv.score(attribute, X_test_embeddings, y_test)
    print("Score:", score)

    predictions = st_model_ridge_cv.predict(attribute, X_test_embeddings)
    predictions = [round_border_score(p) for p in predictions]

    print(f"Prediction samples ({attribute}) (prediction / label):")
    for j in range(5):
        print(predictions[j], y_test[j])

    if predictions_df.columns.empty:
        predictions_df["text_id"] = test_df["text_id"]
        predictions_df[dataset_text_attribute] = test_df[dataset_text_attribute]

    predictions_df[attribute] = test_df[attribute]
    predictions_df[f"{attribute}_predictions"] = predictions

    mcrmse_calculator = MCRMSECalculator()
    mcrmse_calculator.compute_column(y_test, predictions)
    score = mcrmse_calculator.get_score()
    print(f"MCRMSE ({attribute}):", score)


    del X_test_embeddings
    del X_train_embeddings
    # Clear embeddings from memory
    torch.cuda.empty_cache()

    return score
