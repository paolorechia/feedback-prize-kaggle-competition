import re
import sys
from abc import abstractmethod
from datetime import datetime
from uuid import uuid4

import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline,
    set_seed,
)

from load_data import create_train_test_df
from utils import attributes


class TextGenerator:
    @abstractmethod
    def generate(self, text: str, max_length: int) -> str:
        raise NotImplementedError


class GPTNeoGenerator(TextGenerator):
    def __init__(
        self,
        model_path="/data/dropout_test/EleutherAI/gpt-neo-1.3B",
        device="cuda:0",
        do_sample=True,
        temperature=0.9,
    ):
        self.generator = pipeline(
            "text-generation",
            model=model_path,
            device=device,
        )
        self.do_sample = do_sample
        self.temperature = temperature

    def generate(
        self,
        text: str,
        max_length: int,
        do_sample: bool = None,
        temperature: float = None,
    ) -> str:
        if do_sample is None:
            do_sample = self.do_sample
        if temperature is None:
            temperature = self.temperature

        return (
            self.generator(
                text,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )[0]["generated_text"],
            None,
        )


class T0Generator(TextGenerator):
    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

        self.tokenizer = tokenizer
        self.model = model

    def generate(self, text: str, *args, **kwargs) -> str:
        inputs = self.tokenizer.encode(
            text,
            return_tensors="pt",
        )
        return self.model.generate(inputs), None


class GPT2Generator(TextGenerator):
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2",
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
        ).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def generate(
        self,
        text: str,
        max_length: int,
        do_sample=True,
        temperature=0.9,
    ) -> str:
        if do_sample is None:
            do_sample = self.do_sample
        if temperature is None:
            temperature = self.temperature

        input_ids = self.tokenizer.encode(text, return_tensors="pt").cuda()
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
        )
        # print(outputs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text, outputs


def get_low_quality_df():
    """
    This function returns a dataframe with all the low quality essays.
    :return: A dataframe with all the low quality essays.
    """
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
            # low_syntax,
            # low_grammar,
            # low_conventions,
            # low_phraseology,
            # low_vocabulary,
        ]
    )
    return low_quality


def add_labels_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add labels to a dataframe.

    :param df: A dataframe.
    :type df: pandas.DataFrame
    :return: A dataframe with labels.
    :rtype: pandas.DataFrame
    :raises ValueError: If the dataframe is empty.
    """

    labels = []
    for _, row in df.iterrows():
        label = f"(on a scale of 1.0 to 5.0) {[(attribute, row[attribute]) for attribute in attributes]}\n\n"
        labels.append(label)

    df["labels"] = labels


def generate_from_df(
    input_df: pd.DataFrame, text_generator: TextGenerator, reused_length=256
) -> pd.DataFrame:

    generated_texts = []

    output = pd.DataFrame()

    net_outputs = []
    for _, row in input_df.iterrows():
        full_text = row["full_text"]
        # Remove all repeated whitespaces
        text = re.sub(r"\s+", " ", full_text)[:reused_length]
        prompt_instruction = "{}".format(text)

        prompt = "{}{}".format(prompt_instruction, text)
        generated_texts = []

        max_length = len(text) * 2
        # print(len(text), max_length)
        # print(prompt)
        generated_text, net_output = text_generator.generate(prompt, max_length)
        net_outputs.append(net_output)
        # print("Generated Text", generated_text)
        generated_text = generated_text[(len(prompt_instruction) + len(text)) :]
        generated_text = re.sub(r"\s+", " ", generated_text)

        if len(generated_text) > 0:
            generated_texts.append(generated_text)

            output = output.append(
                {
                    "text_id": str(uuid4()),
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
    return output, net_outputs


if __name__ == "__main__":

    if len(sys.argv) > 1:
        generation_batches = int(sys.argv[1])
    else:
        generation_batches = 100

    low_quality_df = get_low_quality_df()
    add_labels_to_df(low_quality_df)

    print("Using GPT-Neo with {} batches".format(generation_batches))
    print("Initializing GPT-Neo generator")
    generator = GPTNeoGenerator()
    for i in range(generation_batches):
        print(f"Starting batch {i}")
        print("Getting low quality essays")

        reused_length = 1024

        print("Generating new essays")
        output_df, _ = generate_from_df(low_quality_df, generator, reused_length)
        output_df.to_csv(
            "generated_csvs/gpt_neo_low_quality_{}_{}.csv".format(
                "full", datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            ),
            index=False,
        )
        print("Done")

        for idx, row in output_df.iterrows():
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
