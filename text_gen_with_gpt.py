from load_data import create_train_test_df
from datetime import datetime
from utils import attributes
import pandas as pd
import sys
import re

from tqdm import tqdm

train_df, test_df = create_train_test_df(0.2, "full")
low_cohesion = train_df[train_df.cohesion == 1.0]
low_syntax = train_df[train_df.syntax == 1.0]
low_grammar = train_df[train_df.grammar == 1.0]
low_conventions = train_df[train_df.conventions == 1.0]
low_phraseology = train_df[train_df.phraseology == 1.0]
low_vocabulary = train_df[train_df.vocabulary == 1.0]

low_quality = pd.concat(
    [
        low_cohesion,
        low_syntax,
        low_grammar,
        low_conventions,
        low_phraseology,
        low_vocabulary,
    ]
)

labels = []
for _, row in low_quality.iterrows():
    label = f"(on a scale of 1.0 to 5.0) {[(attribute, row[attribute]) for attribute in attributes]}\n\n"
    labels.append(label)

low_quality["labels"] = labels

generated_texts = []

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="/data/dropout_test/EleutherAI/gpt-neo-1.3B",
    device="cuda:0",
)


reused_length = 256


def gen_text(prompt: str, max_length) -> str:

    return generator(prompt, max_length=max_length, do_sample=True, temperature=0.9)[0][
        "generated_text"
    ]


output = pd.DataFrame()

for _, row in tqdm(iterable=low_quality.iterrows(), total=len(low_quality)):
    full_text = row["full_text"]
    # Remove all repeated whitespaces
    text = re.sub(r"\s+", " ", full_text)[:reused_length]

    prompt_instruction = "You're a highschool student, writing an essay. Please continue it in the same style, subject and quality: \nEssay: "
    prompt = "{}{}".format(prompt_instruction, text)
    generated_texts = []

    max_length = len(text) * 2
    print(len(text), max_length)
    print(prompt)
    generated_text = gen_text(prompt, max_length)[
        (len(prompt_instruction) + len(text)) :
    ]
    generated_text = re.sub(r"\s+", " ", generated_text)
    generated_texts.append(generated_text)

    # generated_texts.append(generated_text.split("ESSAY_END")[1])
    for t in generated_texts:
        output = output.append(
            {
                "full_text": generated_text,
                "cohesion": row["cohesion"],
                "syntax": row["syntax"],
                "vocabulary": row["vocabulary"],
                "grammar": row["grammar"],
                "conventions": row["conventions"],
                "phraseology": row["phraseology"],
                "labels": row["labels"],
                "source_text": text,
            },
            ignore_index=True,
        )

for idx, row in output.iterrows():
    print(f"TEXT: {idx} ===========================================")
    print(row["labels"])

    print("ORIGINAL TEXT")
    text = re.sub(r"\s+", " ", row["source_text"])
    print(text[0:reused_length])
    print("===========================================")

    print("GENERATED TEXT")
    print(row["full_text"])
    print("===========================================")
    print("===========================================")
    print("===========================================")


output.to_csv("generated_csvs/generated_low_quality.csv", index=False)
