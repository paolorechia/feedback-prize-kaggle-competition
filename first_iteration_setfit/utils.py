import pandas as pd
from tqdm import tqdm

minimum_chunk_length = 10
setfit_model_max_length = 256

attributes = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]


labels = {
    "1.0": "terrible",
    "1.5": "bad",
    "2.0": "poor",
    "2.5": "fair",
    "3.0": "average",
    "3.5": "good",
    "4.0": "great",
    "4.5": "excellent",
    "5.0": "perfect",
}
reverse_labels = {v: float(k) for k, v in labels.items()}
reverse_labels["average_or_below_average"] = 1.75
reverse_labels["above_average"] = 3.75


def fit_float_score_to_nearest_valid_point(float_score: float):
    """Fit float score to nearest valid point."""
    valid_points = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    return min(valid_points, key=lambda x: abs(x - float_score))


def round_border_score(float_score: float):
    if float_score < 1.0:
        return 1.0
    if float_score > 5.0:
        return 5.0
    return float_score


def test_fit_float_score_to_nearest_valid_point():
    """Tests function"""
    assert fit_float_score_to_nearest_valid_point(0.0) == 1.0
    assert fit_float_score_to_nearest_valid_point(0.9) == 1.0
    assert fit_float_score_to_nearest_valid_point(1.1) == 1.0
    assert fit_float_score_to_nearest_valid_point(1.24) == 1.0
    assert fit_float_score_to_nearest_valid_point(1.27) == 1.5
    assert fit_float_score_to_nearest_valid_point(1.74) == 1.5
    assert fit_float_score_to_nearest_valid_point(1.76) == 2.0
    assert fit_float_score_to_nearest_valid_point(2.26) == 2.5
    assert fit_float_score_to_nearest_valid_point(2.76) == 3.0
    assert fit_float_score_to_nearest_valid_point(3.3) == 3.5
    assert fit_float_score_to_nearest_valid_point(3.6) == 3.5
    assert fit_float_score_to_nearest_valid_point(3.76) == 4.0
    assert fit_float_score_to_nearest_valid_point(4.2) == 4.0
    assert fit_float_score_to_nearest_valid_point(4.3) == 4.5
    assert fit_float_score_to_nearest_valid_point(4.6) == 4.5
    assert fit_float_score_to_nearest_valid_point(4.76) == 5.0
    assert fit_float_score_to_nearest_valid_point(5.0) == 5.0
    assert fit_float_score_to_nearest_valid_point(10.0) == 5.0


class MCRMSECalculator:
    def __init__(self):
        self._sum = 0.0
        self._samples = 0

    def compute_score_for_df(self, df):
        for index, row in df.iterrows():
            inner_sum = 0.0
            for attribute in attributes:
                inner_sum += (row[attribute] - row[f"{attribute}_predictions"]) ** 2
            inner_sum /= len(attributes)
            self._sum += inner_sum
            self._samples += 1

    def compute_column(self, labels, predictions):
        points = zip(labels, predictions)
        column_sum = 0.0
        for point in points:
            column_sum += (point[0] - point[1]) ** 2
        self._sum += column_sum / len(labels)
        self._samples += 1

    def get_score(self):
        return self._sum / self._samples


def break_sentences(
    train_df, setfit_model_max_length, minimum_chunk_length, binary_label=None
) -> pd.DataFrame:
    assert train_df.isna().values.any() == False

    broken_sentences = pd.DataFrame()
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

    print("Chunking sentences...")
    print("Length of df: ", len(train_df))
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
                data = [
                    row["text_id"],
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
                if binary_label:
                    data.append(row[binary_label])

                new_sentence_row = pd.DataFrame(
                    columns=train_df.columns,
                    data=[data],
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
    A = train_df[train_df.too_long == False]
    A = A[A.too_short == False]
    merged_df = pd.concat(
        [
            A,
            broken_sentences,
        ]
    )
    print("After merging chunked sentences: ")
    print(len(merged_df))
    # print("Too long values")
    assert merged_df.too_long.values.any() == False
    # print(merged_df[merged_df.too_long == True].sentence_length.describe())

    print("Too short values")
    assert merged_df.too_short.values.any() == False
    # print(merged_df[merged_df.too_short == True].sentence_length.describe())
    # print(merged_df.cohesion.unique())

    return merged_df


def split_df_into_sentences(train_df: pd.DataFrame, binary_label=None) -> pd.DataFrame:
    def split_text_into_sentences(text):
        sentences = text.split(".")
        return sentences

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
        sentences = split_text_into_sentences(text)
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