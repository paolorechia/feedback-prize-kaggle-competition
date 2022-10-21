import os
import pandas as pd

intermediary_csvs = "./intermediary_csvs"

from pre_trained_st_model import MultiHeadSentenceTransformerModelRidgeCV
from model_catalog import ModelCatalog

# We load test as train on purpose, to use test-size as 20%
train_df = pd.read_csv(os.path.join(intermediary_csvs, "testcohesion_0.8_256_64.csv"))
test_df = pd.read_csv(os.path.join(intermediary_csvs, "train_cohesion_0.8_256_64.csv"))

model_info = ModelCatalog.DebertaV3
model = MultiHeadSentenceTransformerModelRidgeCV(
    model=model_info.model_name,
)

train_df