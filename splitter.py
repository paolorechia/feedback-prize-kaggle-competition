from abc import abstractmethod
import pandas as pd
from tqdm import tqdm
from scipy import stats
from statistics import (
    fmean,
    median,
)
import numpy as np
from utils import attributes
from dataclasses import dataclass


def _split_text_into_sentences(text):
    sentences = text.split(".")
    return sentences


def split_text_into_n_parts(text, n, minimum_chunk_length):
    part_length = int(len(text) // n)
    sentences = []
    for i in range(0, len(text), part_length):
        part = text[i : i + part_length]
        if len(part.strip()) > minimum_chunk_length:
            sentences.append(part)
    return sentences


def split_text_into_sliding_windows(text, window_size, step_size, minimum_chunk_length):
    sentences = []
    for i in range(0, len(text), step_size):
        part = text[i : i + window_size]
        if len(part.strip()) > minimum_chunk_length:
            sentences.append(part)
    return sentences


def split_text_into_half(text):
    half_len = len(text) // 2
    sentences = [text[0:half_len], text[half_len:]]
    return sentences


def split_df_into_sentences(
    train_df: pd.DataFrame,
    sentence_function=_split_text_into_sentences,
    binary_label=None,
) -> pd.DataFrame:

    new_columns = [
        "text_id",
        "sentence_text",
        "sentence_length",
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]
    if binary_label:
        new_columns.append(binary_label)

    # iterate over each row in the dataframe
    sentence_train_dataframe = pd.DataFrame(columns=new_columns)

    print("Length of df: ", len(train_df))
    for index, row in tqdm(iterable=train_df.iterrows(), total=len(train_df)):
        # get the text
        text = row["full_text"]
        # split the text into sentences
        sentences = sentence_function(text)
        # iterate over each sentence
        for sentence in sentences:
            # create a new row in the dataframe
            if len(sentence) > 0:
                data = [
                    row["text_id"],
                    sentence,
                    len(sentence),
                    row["cohesion"],
                    row["syntax"],
                    row["vocabulary"],
                    row["phraseology"],
                    row["grammar"],
                    row["conventions"],
                ]
                if binary_label:
                    data.append(row[binary_label])
                sentence_train_dataframe = pd.concat(
                    [
                        sentence_train_dataframe,
                        pd.DataFrame(
                            columns=new_columns,
                            data=[data],
                        ),
                    ]
                )
    return sentence_train_dataframe


def unroll_sentence_df(
    sentence_df, embeddings, attribute, train_max_length=0, trained_model=None
):
    unrolled = []
    texts = {}
    max_length = 0
    for text_id, embedding, attribute_value in zip(
        sentence_df["text_id"], embeddings, sentence_df[attribute]
    ):
        if text_id not in texts:
            texts[text_id] = {
                "embeddings": [],
                "attributes": [],
            }
        texts[text_id]["embeddings"].extend(embedding)
        if trained_model:
            predicted_attribute = trained_model.predict(
                f"{attribute}_embeddings", [embedding]
            )[0]
            texts[text_id]["attributes"].append(predicted_attribute)
        else:
            texts[text_id]["attributes"].append(attribute_value)
        max_length = max(len(texts[text_id]["embeddings"]), max_length)

    if train_max_length > 0:
        safe_guard = train_max_length
    else:
        safe_guard = max_length * 4
    if safe_guard < max_length:
        raise ValueError(
            "Max length of test set is larger than train set, cannot fit model."
        )
    for text_id, text in texts.items():
        if len(text["embeddings"]) < safe_guard:
            text["embeddings"].extend([0] * (safe_guard - len(text["embeddings"])))

        unrolled.append(
            {
                "text_id": text_id,
                "embeddings": text["embeddings"],
                "attributes": text["attributes"],
                "features": []
                + np.quantile(
                    text["attributes"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                ).tolist()
                + [sum(text["attributes"]) / len(text["attributes"])]
                + [max(text["attributes"])]
                + [min(text["attributes"])]
                + [len(text["attributes"])]
                + [median(text["attributes"])]
                + [fmean(text["attributes"])]
                + [stats.gmean(text["attributes"])]
                + [stats.kurtosis(text["attributes"])]
                + [stats.skew(text["attributes"])]
                + [stats.moment(text["attributes"], moment=1)]
                + [stats.moment(text["attributes"], moment=2)]
                + [stats.moment(text["attributes"], moment=3)]
                + [stats.moment(text["attributes"], moment=4)],
            }
        )

    unrolled_df = pd.DataFrame(unrolled)
    return unrolled_df, safe_guard


@dataclass
class Labels:
    cohesion: float
    syntax: float
    vocabulary: float
    phraseology: float
    grammar: float
    conventions: float


def unroll_labelled_sentence_df_all(sentence_df, embeddings):
    texts = {}
    max_length = 0

    labels_list = []
    for _, row in sentence_df.iterrows():
        labels_list.append(
            Labels(
                cohesion=row["cohesion"],
                syntax=row["syntax"],
                vocabulary=row["vocabulary"],
                phraseology=row["phraseology"],
                grammar=row["grammar"],
                conventions=row["conventions"],
            )
        )

    iterator = zip(sentence_df["text_id"], embeddings, labels_list)
    for (text_id, embedding, row_labels) in iterator:
        if text_id not in texts:
            texts[text_id] = {
                "embeddings": [],
                "attributes": {k: [] for k in attributes},
            }
        texts[text_id]["embeddings"].extend(embedding)
        max_length = max(len(texts[text_id]["embeddings"]), max_length)
        for attr in attributes:
            texts[text_id]["attributes"][attr].append(getattr(row_labels, attr))

    safe_guard = max_length
    return _create_unrolled_df_with_labels(texts, safe_guard)


def infer_labels(
    sentence_df,
    embeddings,
    trained_model,
    trained_length,
    attribute=None,
):
    # Inference flow
    iterator = zip(sentence_df["text_id"], embeddings)
    texts = {}
    for (text_id, embedding) in iterator:
        if text_id not in texts:
            texts[text_id] = {
                "embeddings": [],
                "attributes": {k: [] for k in attributes},
            }
        texts[text_id]["embeddings"].extend(embedding)

        for key, array in texts[text_id]["attributes"].items():
            if attribute is not None:
                if key != attribute:
                    continue

            predicted_attribute = trained_model.predict(
                f"{key}_embeddings", [embedding]
            )[0]
            array.append(predicted_attribute)

    unrolled_df, _ = _create_unrolled_df_with_labels(texts, trained_length, attribute)
    return unrolled_df


def _create_unrolled_df_with_labels(texts, safe_guard, attribute=None):
    unrolled = []

    for text_id, text in texts.items():
        if len(text["embeddings"]) < safe_guard:
            text["embeddings"].extend([0] * (safe_guard - len(text["embeddings"])))

        if len(text["embeddings"]) > safe_guard:
            raise ValueError("Text is too long for the model.")

        unrolled_row = {
            "text_id": text_id,
            "embeddings": np.array(text["embeddings"]),
        }
        for attr in attributes:
            if attribute is not None:
                if attr != attribute:
                    continue
            unrolled_row[attr] = text["attributes"][attr]

        for label_name in text["attributes"].keys():
            if attribute is not None:
                if label_name != attribute:
                    continue

            attr = text["attributes"][label_name]
            unrolled_row[f"{label_name}_features"] = (
                []
                + [sum(attr) / len(attr)]
                + [max(attr)]
                + [min(attr)]
                + [len(attr)]
                + np.quantile(
                    text["attributes"][label_name],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                ).tolist()
                + [median(attr)]
                + [stats.gmean(attr)]
                + [stats.kurtosis(attr)]
                + [stats.skew(attr)]
                + [stats.moment(attr, moment=1)]
                + [stats.moment(attr, moment=2)]
                + [stats.moment(attr, moment=3)]
                + [stats.moment(attr, moment=4)]
            )
        unrolled.append(unrolled_row)
    unrolled_df = pd.DataFrame(unrolled)
    return unrolled_df, safe_guard


class SplittingStrategy:
    def __init__(self, splitter, name):
        self.splitter = splitter
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Encoder:
    @abstractmethod
    def encode(self, text):
        raise NotImplementedError()


def smart_blockenizer(
    input_df: pd.DataFrame,
    sentence_csv_dir: str,
    columns_mapping,
    multi_head: Encoder,
    splitting_strategy: SplittingStrategy,
):

    sentence_df_path = (
        f"{sentence_csv_dir}/full_optimized_{splitting_strategy.name}.csv"
    )
    try:
        full_sentence_df = pd.read_csv(sentence_df_path)
    except Exception:
        full_sentence_df = split_df_into_sentences(
            input_df, splitting_strategy.splitter
        )
        full_sentence_df.to_csv(sentence_df_path, index=False)

    print("Length of full_sentence_df", len(full_sentence_df))
    print(full_sentence_df)

    cm = columns_mapping
    embeddings = [
        np.array(e)
        for e in multi_head.encode(
            full_sentence_df[cm["text"]],
            batch_size=32,
            show_progress_bar=True,
            use_cache=f"full-dataframe-optimized-sentence-encoding-{splitting_strategy.name}",
        )
    ]
    full_sentence_df["embeddings"] = embeddings
    model_embedding_length = len(embeddings[0])

    print("Model embedding length", model_embedding_length)

    # Make each row in full sentence DF become a new column in the full df
    subset = list(
        zip(
            full_sentence_df[cm["id"]],
            full_sentence_df[cm["text"]],
            full_sentence_df["embeddings"],
        )
    )

    blocks_dict = {}

    num_blocks = 0
    for tuple_ in subset:
        text_id = tuple_[0]
        sentence = tuple_[1]
        embeddings = tuple_[2]
        if text_id not in blocks_dict:
            blocks_dict[text_id] = []
            blocks_dict[text_id].append(
                {"sentence": sentence, "embeddings": embeddings}
            )
        else:
            blocks_dict[text_id].append(
                {"sentence": sentence, "embeddings": embeddings}
            )
        num_blocks = max(num_blocks, len(blocks_dict[text_id]))

    block_columns = []
    block_embeddings = []
    n_blocks_column = []
    for j in range(num_blocks):
        block_columns.append([])
        block_embeddings.append([])

    required_length = num_blocks

    for idx, row in input_df.iterrows():
        text_id = row[cm["id"]]
        blocks_ = blocks_dict[text_id]
        n_blocks_column.append(len(blocks_))
        # Pad with empty strings
        while len(blocks_) < required_length:
            blocks_.append(
                {"sentence": "", "embeddings": np.zeros(model_embedding_length)}
            )

        for j in range(num_blocks):
            block_columns[j].append(blocks_[j]["sentence"])
            block_embeddings[j].append(blocks_[j]["embeddings"])

    input_df["number_blocks"] = n_blocks_column
    for j in range(num_blocks):
        input_df[f"block_{j}"] = block_columns[j]
        input_df[f"embeddings_{j}"] = block_embeddings[j]
