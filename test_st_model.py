
from os import lseek
from sklearn.linear_model import RidgeCV
from sentence_transformers import SentenceTransformer
from load_data import create_attribute_stratified_split


train_df, test_df = create_attribute_stratified_split("cohesion", 0.2, "sampled")

X_train = train_df["full_text"]
y_train = train_df["cohesion"]

X_test = test_df["full_text"]
y_test = test_df["cohesion"]

st_model = SentenceTransformer("./st_output/all-distilroberta-v1")

print("Encoding training set")
X_train_embeddings = st_model.encode(X_train)
ridge_cv = RidgeCV()
print("Fitting Ridge CV...")
ridge_cv.fit(X_train_embeddings, y_train)

print("Encoding test set")
X_test_embeddings = st_model.encode(X_test)

print("Scoring...")
score = ridge_cv.score(X_test_embeddings, y_test)

print(score)