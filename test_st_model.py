import os
from pre_trained_st_model import SentenceTransformerModelRidgeCV
from utils import MCRMSECalculator, attributes
from load_data import create_attribute_stratified_split
from model_catalog import ModelCatalog

attribute = "cohesion"
experiment_id = "3eee1504"
test_size_from_experiment = 0.5
base_trained_model_folder = "/data/feedback-prize/st-output/"
experiment_dataset = "full"
dataset_text_attribute = "full_text"
is_multi_task = True

# model_info = ModelCatalog.AllMpnetBasev1
# model_info = ModelCatalog.BertBaseUncased
# model_info = ModelCatalog.DebertaV3
model_info = ModelCatalog.DebertaV3Large

model_name = model_info.model_name


def test_on_attribute(
    attribute: str,
    st_model_ridge_cv: SentenceTransformerModelRidgeCV,
):
    print("Testing on attribute: ", attribute)
    train_df, test_df = create_attribute_stratified_split(
        attribute, test_size_from_experiment, experiment_dataset
    )

    X_train = list(train_df[dataset_text_attribute])
    y_train = list(train_df[attribute])

    X_test = list(test_df[dataset_text_attribute])
    y_test = list(test_df[attribute])

    st_model_ridge_cv.fit(X_train, y_train)
    score = st_model_ridge_cv.score(X_test, y_test)
    print("Score:", score)

    predictions = st_model_ridge_cv.predict(X_test)
    print("Prediction samples (prediction / label):")
    for j in range(5):
        print(predictions[j], y_test[j])
    mcrmse_calculator = MCRMSECalculator()
    mcrmse_calculator.compute_column(y_test, predictions)
    print("MCRMSE:", mcrmse_calculator.get_score())


model_folder = f"{model_name}-{attribute}-{experiment_id}"
model_path = os.path.join(base_trained_model_folder, model_folder)


if not is_multi_task:
    st_model_ridge_cv = SentenceTransformerModelRidgeCV(model_path)
    test_on_attribute(attribute, st_model_ridge_cv)
else:
    print("Testing multi task on all attributes")
    model_path = (
        "/data/feedback-prize/st-output/microsoft/deberta-v3-base-multitask-66955b1c"
    )
    st_model_ridge_cv = SentenceTransformerModelRidgeCV(model_path)
    for attribute in attributes:
        test_on_attribute(attribute, st_model_ridge_cv)
