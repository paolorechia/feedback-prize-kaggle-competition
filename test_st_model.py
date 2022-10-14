import os
from pre_trained_st_model import SentenceTransformerModelRidgeCV
from utils import MCRMSECalculator
from load_data import create_attribute_stratified_split
from model_catalog import ModelCatalog

attribute = "cohesion"
experiment_id = "3eee1504"
test_size_from_experiment = 0.5
base_trained_model_folder = "/data/feedback-prize/st-output/"
experiment_dataset = "full"
dataset_text_attribute = "full_text"

# model_info = ModelCatalog.AllMpnetBasev1
# model_info = ModelCatalog.BertBaseUncased
# model_info = ModelCatalog.DebertaV3
model_info = ModelCatalog.DebertaV3Large

model_name = model_info.model_name

train_df, test_df = create_attribute_stratified_split(
    attribute, test_size_from_experiment, experiment_dataset
)

X_train = list(train_df[dataset_text_attribute])
y_train = list(train_df[attribute])

X_test = list(test_df[dataset_text_attribute])
y_test = list(test_df[attribute])


model_folder = f"{model_name}-{attribute}-{experiment_id}"
model_path = os.path.join(base_trained_model_folder, model_folder)

# Load from checkpoint instead
# model_path = "/data/feedback-prize/st-checkpoints/all-distilroberta-v1/cohesion/1c524550/50"

st_model_ridge_cv = SentenceTransformerModelRidgeCV(model_path)
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
