import pandas as pd
from matplotlib import pyplot as plt

train_df = pd.read_csv("/data/feedback-prize/sentence_chunked_train.csv")
print(len(train_df))
print(train_df.sentence_length.describe())
print(train_df.sentence_length.value_counts())

train_df.sentence_length.hist(bins=100)
print(train_df.isna().values)
plt.show()