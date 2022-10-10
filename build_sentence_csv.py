import pandas as pd
from utils import split_df_into_sentences

train_df = pd.read_csv("/data/feedback-prize/train.csv")
sentence_train_dataframe = split_df_into_sentences(train_df)
sentence_train_dataframe.to_csv("/data/feedback-prize/sentence_train.csv", index=False)
