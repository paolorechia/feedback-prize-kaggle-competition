import os
import gzip
import json
from utils import remove_repeated_whitespaces


def load_bbc_news(head=3000):
    filepath = "/data/text-datasets/bbc"
    dirs_ = os.listdir(filepath)
    data = []
    count = 0
    for dir_ in dirs_:
        files = os.listdir(os.path.join(filepath, dir_))
        for file in files:
            if count > head:
                break
            with open(os.path.join(filepath, dir_, file), "r") as f:
                try:
                    text = f.read()
                    data.append({"review_text": text, "label": dir_})
                    count += 1
                except UnicodeDecodeError:
                    pass
    return data


def load_amazon_reviews(head=3000):
    filepath_1 = "/data/text-datasets/amazon_review/train.ft.txt"
    filepath_2 = "/data/text-datasets/amazon_review/test.ft.txt"
    data = []
    count = 0
    for filepath in [filepath_1, filepath_2]:
        with open(filepath, "r") as f:
            for line in f:
                if count > head:
                    break
                line = line.strip()
                label, text = line.split(" ", 1)
                text = remove_repeated_whitespaces(text)
                data.append({"review_text": text, "label": label})
                count += 1
    return data


def load_steam_reviews(head=3000):
    filepath = "/data/text-datasets/steam_reviews.json.gz"
    g = gzip.open(filepath, "r")
    data = []
    count = 0
    for l in g:
        if count > head:
            break
        d = eval(l)
        d["review_text"] = d["text"]
        del d["text"]
        data.append(d)
        count += 1
    return data


def load_goodreads_reviews(head=3000):
    filepath = "/data/text-datasets/goodreads_reviews_dedup.json.gz"
    count = 0
    data = []
    with gzip.open(filepath) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)

            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data
