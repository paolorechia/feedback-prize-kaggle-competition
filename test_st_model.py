import os
from pre_trained_st_model import SentenceTransformerModelRidgeCV
from utils import MCRMSECalculator
from load_data import create_attribute_stratified_split


train_df, test_df = create_attribute_stratified_split("cohesion", 0.5, "full")

X_train = list(train_df["full_text"])
y_train = list(train_df["cohesion"])

X_test = list(test_df["full_text"])
y_test = list(test_df["cohesion"])


base_trained_model_folder = "/data/feedback-prize/st-output/"
model_folder = "all-distilroberta-v1-cohesion-604f20b4"
model_path = os.path.join(base_trained_model_folder, model_folder)

st_model_ridge_cv = SentenceTransformerModelRidgeCV(model_path)
st_model_ridge_cv.fit(X_train, y_train)
score = st_model_ridge_cv.score(X_test, y_test)
print("Score:", score)

predictions = st_model_ridge_cv.predict(X_test)
mcrmse_calculator = MCRMSECalculator()
mcrmse_calculator.compute_column(y_test, predictions)
print("MCRMSE:", mcrmse_calculator.get_score())
