import pandas as pd
from utils import attributes, round_border_score, MCRMSECalculator
from pre_trained_st_model import MultiHeadSentenceTransformerModelRidgeCV
from sentence_transformers import SentenceTransformer
from load_data import create_attribute_stratified_split


def evaluate_mcrmse_multitask(
    dataset_text_attribute: str,
    test_size_from_experiment: float,
    input_dataset: str,
    st_model: SentenceTransformer,
):

    if not isinstance(st_model, SentenceTransformer):
        raise ValueError("Expected SentenceTransformer")

    st_model_ridge_cv = MultiHeadSentenceTransformerModelRidgeCV(st_model)

    predictions_df = pd.DataFrame()

    print("Evaluating MCRMSE on multitask...")
    for attribute in attributes:
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

        X_train_embeddings = st_model.encode(X_train)
        X_test_embeddings = st_model.encode(X_test)

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
        print(f"MCRMSE ({attribute}):", mcrmse_calculator.get_score())
        predictions_df[attribute] = predictions

    # Compute MCRMSE for all attributes
    # Merge each predictions_df with the previous one
    # deduplicating the text_id column and averaging the scores
    # for the same text_id
    predictions_df = predictions_df.groupby("text_id").mean().reset_index()
    mcrmse_calculator = MCRMSECalculator()
    mcrmse_calculator.compute_score_for_df(predictions_df)
    print("MCRMSE (all attributes):", mcrmse_calculator.get_score())
