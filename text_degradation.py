import string
import random
import pandas as pd

def _degradate_by_random_typos(text, p=0.1):
    text = list(text)
    for i in range(len(text)):
        if random.random() < p:
            text[i] = random.choice(string.ascii_lowercase)
    return "".join(text)


def _degradate_by_deleting_random_word(text, p=0.1):
    text = text.split()
    for i in range(len(text)):
        if random.random() < p:
            text[i] = ""
    return " ".join(text)


def _degradate_by_deleting_random_char(text, p=0.1):
    text = list(text)
    for i in range(len(text)):
        if random.random() < p:
            text[i] = ""
    return "".join(text)


def _degradate_by_changing_word_order(text):
    text = text.split()
    if len(text) <= 1:
        return text

    number_of_swapped_words = random.randint(1, len(text) // 2)
    for _ in range(number_of_swapped_words):
        i = random.randint(0, len(text) - 1)
        j = random.randint(0, len(text) - 1)
        text[i], text[j] = text[j], text[i]
    return " ".join(text)


def degradate_text(text):
    if random.random() < 0.25:
        text = _degradate_by_random_typos(text)
    if random.random() < 0.25:
        text = _degradate_by_deleting_random_word(text)
    if random.random() < 0.25:
        text = _degradate_by_deleting_random_char(text)
    if random.random() < 0.25:
        text = _degradate_by_changing_word_order(text)
    return text


def degradate_df_text(
    df: pd.DataFrame,
    text_label: str,
    degradation_rate: float,
    minimum_text_length: int = 64,
):
    new_df = pd.DataFrame()
    texts = df[text_label]
    new_texts = []
    for text in texts:
        if random.random() > degradation_rate:
            new_texts.append(text)
        else:
            degradated_text = degradate_text(text)
            if len(degradated_text) <= minimum_text_length:
                continue
            new_texts.append(degradated_text)
    new_df[text_label] = new_texts
    return new_df
