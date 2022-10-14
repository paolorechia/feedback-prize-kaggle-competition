
from pre_trained_st_model import SentenceTransformerModelRidgeCV
from load_data import create_attribute_stratified_split


train_df, test_df = create_attribute_stratified_split("cohesion", 0.2, "sampled")

X_train = list(train_df["full_text"])
y_train = list(train_df["cohesion"])

X_test = list(test_df["full_text"])
y_test = list(test_df["cohesion"])

st_model_ridge_cv = SentenceTransformerModelRidgeCV("./st_output/all-distilroberta-v1")
st_model_ridge_cv.fit(X_train, y_train)
score = st_model_ridge_cv.score(X_test, y_test)
print("Score:", score)