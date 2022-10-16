import pandas as pd
train_df = pd.read_csv("/data/feedback-prize/train.csv")
train_df["text_length"] = train_df.apply(lambda x: len(x.full_text), axis=1)
desc = train_df["text_length"].describe()
print(desc)

