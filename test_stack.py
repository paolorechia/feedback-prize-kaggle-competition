from load_data import create_train_test_df
from utils import attributes
from model_stacker import ModelStack
from model_catalog import ModelCatalog
from pre_trained_st_model import MultiHeadSentenceTransformerModelRidgeCV
from utils import calculate_rmse_score
import pandas as pd

train_df, test_df = create_train_test_df(test_size=0.2, dataset="full")

stack = ModelStack(
    [
        ModelCatalog.DebertaV3,
    ]
)

X_train = list(train_df["full_text"])
X_test = list(test_df["full_text"])


multi_head = MultiHeadSentenceTransformerModelRidgeCV(stack)

X_train_embeddings = multi_head.encode(X_train, batch_size=32)
print(X_train_embeddings.shape)
X_test_embeddings = multi_head.encode(X_test, batch_size=32)
print(X_test_embeddings.shape)

preds_df = pd.DataFrame()
preds_df["text_id"] = test_df["text_id"]
preds_df["full_text"] = test_df["full_text"]

for attribute in attributes:
    print("Evaluating on attribute: ", attribute)
    multi_head.fit(attribute, X_train_embeddings, train_df[attribute])
    s = multi_head.score(attribute, X_test_embeddings, test_df[attribute])
    print("Regressor Score:", s)
    preds = multi_head.predict(attribute, X_test_embeddings)
    preds_df[attribute] = preds

score = calculate_rmse_score(test_df[attributes].values, preds_df[attributes].values)
print("RMSE Score:", score)
