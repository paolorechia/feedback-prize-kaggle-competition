import pandas as pd
from tqdm import tqdm

train_df = pd.read_csv("/data/feedback-prize/train.csv")


def split_text_into_sentences(text):
    sentences = text.split(".")
    return sentences


new_columns = [
    "sentence_text",
    "sentence_length",
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]
# iterate over each row in the dataframe
sentence_train_dataframe = pd.DataFrame(columns=new_columns)

print("Length of train_df: ", len(train_df))
for index, row in tqdm(train_df.iterrows()):
    # get the text
    text = row["full_text"]
    # split the text into sentences
    sentences = split_text_into_sentences(text)
    # iterate over each sentence
    for sentence in sentences:
        # create a new row in the dataframe
        if len(sentence) > 0:
            sentence_train_dataframe = pd.concat(
                [
                    sentence_train_dataframe,
                    pd.DataFrame(
                        columns=new_columns,
                        data=[
                            [
                                sentence,
                                len(sentence),
                                row["cohesion"],
                                row["syntax"],
                                row["vocabulary"],
                                row["phraseology"],
                                row["grammar"],
                                row["conventions"],
                            ]
                        ],
                    ),
                ]
            )

sentence_train_dataframe.to_csv("/data/feedback-prize/sentence_train.csv", index=False)
