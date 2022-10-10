from cmath import isnan
import pandas as pd
import math
from tqdm import tqdm

train_df = pd.read_csv("/data/feedback-prize/sentence_train.csv")

# print(train_df.head())
# Assert that are no NaNs in the dataframe
assert train_df.isna().values.any() == False

setfit_model_max_length = 256
minimum_chunk_length = 32

train_df["sentence_length"] = train_df.apply(lambda x: len(x.sentence_text), axis=1)
train_df["too_long"] = train_df.apply(
    lambda x: x.sentence_length > setfit_model_max_length, axis=1
)
train_df["too_short"] = train_df.apply(
    lambda x: x.sentence_length <= minimum_chunk_length, axis=1
)
print("Prior to chunking: ")
print(train_df["too_long"].value_counts())
print(train_df[train_df.too_long == True].sentence_length.describe())
print(train_df["too_short"].value_counts())
print(train_df[train_df.too_short == True].sentence_length.describe())
print("Cohesion values:")
print(train_df.cohesion.unique())

broken_sentences = pd.DataFrame()

print("Chunking sentences...")
print("Length of train_df: ", len(train_df))
for index, row in tqdm(
    iterable=train_df[train_df.too_long == True].iterrows(),
    total=len(train_df[train_df.too_long == True]),
):
    sentence_chunks = []
    sentence = row["sentence_text"]
    max_length = setfit_model_max_length
    while len(sentence) > max_length:
        new_chunk = sentence[:max_length]
        if type(new_chunk) == str and len(new_chunk) >= minimum_chunk_length:
            new_sentence_row = pd.DataFrame(
                columns=train_df.columns,
                data=[
                    [
                        new_chunk,
                        len(new_chunk),
                        row["cohesion"],
                        row["syntax"],
                        row["vocabulary"],
                        row["phraseology"],
                        row["grammar"],
                        row["conventions"],
                        False,
                        False,
                    ]
                ],
            )
            broken_sentences = pd.concat([broken_sentences, new_sentence_row])
        sentence = sentence[max_length:]

print("Chunked df: ")
print(broken_sentences.head())

broken_sentences["too_long"] = broken_sentences.apply(
    lambda x: x.sentence_length > setfit_model_max_length, axis=1
)
broken_sentences["too_short"] = broken_sentences.apply(
    lambda x: x.sentence_length <= minimum_chunk_length, axis=1
)
print("Cohesion values:")
print(broken_sentences.cohesion.unique())
# All chunks should respect the length limit
assert broken_sentences.too_long.values.any() == False


# Merge shorter sentences with the new chunks
merged_df = pd.concat(
    [
        train_df[train_df.too_long == False][train_df.too_short == False],
        broken_sentences,
    ]
)
print("After merging chunked sentences: ")
merged_df.to_csv("/data/feedback-prize/sentence_chunked_train.csv", index=False)
print(len(merged_df))
print("Too long values")
assert merged_df.too_long.values.any() == False
print(merged_df[merged_df.too_long == True].sentence_length.describe())

print("Too short values")
assert merged_df.too_short.values.any() == False
print(merged_df[merged_df.too_short == True].sentence_length.describe())
print(merged_df.cohesion.unique())
