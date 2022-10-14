from dataclasses import dataclass
from typing import List, Union
from sentence_transformers import SentenceTransformer, InputExample, losses
from tqdm import tqdm


def print_sentence_pair(sent1, sent2, label):
    print("Text 1: \n", sent1, "\n")
    print("Text 2: \n", sent2, "\n")
    print("Similarity Label:", label, "\n")


@dataclass
class EvaluationDataset:
    sentences1: List[str]
    sentences2: List[str]
    scores: List[float]

    def print_sample(self, sample_size=3):
        for i in range(sample_size):
            print_sentence_pair(self.sentences1[i], self.sentences2[i], self.scores[i])
            print("-" * 80)


@dataclass
class TrainingDataset:
    training_pairs: List[InputExample]

    def print_sample(self, sample_size=5):
        for i in range(sample_size):
            print_sentence_pair(
                self.training_pairs[i].texts[0],
                self.training_pairs[i].texts[1],
                self.training_pairs[i].label,
            )
            print("-" * 80)


class SentencePairState:
    mode: str
    state: Union[TrainingDataset, EvaluationDataset]

    def __init__(self, mode="training") -> None:
        if mode == "training":
            self.mode = mode
            self.state = TrainingDataset(training_pairs=[])
        elif mode == "evaluation":
            self.mode = mode
            self.state = EvaluationDataset(sentences1=[], sentences2=[], scores=[])
        else:
            raise ValueError("mode must be either 'training' or 'evaluation'")

    def add_data(self, sentence1: str, sentence2: str, distance_label: float):
        if self.mode == "training":
            self.state.training_pairs.append(
                InputExample(texts=[sentence1, sentence2], label=distance_label)
            )
        elif self.mode == "evaluation":
            self.state.sentences1.append(sentence1)
            self.state.sentences2.append(sentence2)
            self.state.scores.append(distance_label)
        else:
            raise ValueError("mode must be either 'training' or 'evaluation'")


def create_continuous_sentence_pairs(
    df, text_label, attribute, model_truncate_length, mode="training"
) -> List[InputExample]:
    """Creates training pairs for a SentenceTransformer model.

    This function follows the formats from the documentation.

    However, it does something different than the SetFit library.
    Instead of assuming binary labels, it assumes a continuous label.

    Hopefully, this improves the regression performance.

    In the training mode, creates data like this:

    ```
    # Define your train examples. You need more than just two examples...
    train_examples = [
        InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
        InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3),
    ]
    ```

    In the evaluation mode, data comes in this form:
    ```
    sentences1 = [
        "This list contains the first column",
        "With your sentences",
        "You want your model to evaluate on",
    ]
    sentences2 = [
        "Sentences contains the other column",
        "The evaluator matches sentences1[i] with sentences2[i]",
        "Compute the cosine similarity and compares it to scores[i]",
    ]
    scores = [0.3, 0.6, 0.2]
    ```

    Both formats are abstracted with the SentencePairState class.
    """

    state = SentencePairState(mode)
    for _, row in tqdm(iterable=df.iterrows(), total=len(df)):
        text1 = row[text_label].strip()[0:model_truncate_length]
        label1 = row[attribute]
        for _, row in df.iterrows():
            text2 = row[text_label].strip()[0:model_truncate_length]
            label2 = row[attribute]
            if text1 != text2:
                # Possible labels are:
                # 1.0
                # 1.5
                # 2.0
                # 2.5
                # 3.0
                # 3.5
                # 4.0
                # 4.5
                # 5.0

                raw_label_distance = abs(label1 - label2)
                # 1.0 is the minimum distance and 0 is the maximum distance.

                # First, normalize difference to 0-1
                normalized_distance = raw_label_distance / 4.0
                # Now, invert the distance

                label_distance = 1 - (normalized_distance / 4)
                # Let's try some examples
                # 1.0 - ((5.0 - 1.0) / 4) = 0.0
                # 1.0 - ((5.0 - 2.0) / 4) = 0.25
                # 1.0 - ((5.0 - 3.0) / 4) = 0.5
                # 1.0 - ((5.0 - 4.0) / 4) = 0.75
                # 1.0 - ((5.0 - 5.0) / 4) = 1.0

                state.add_data(
                    sentence1=text1, sentence2=text2, distance_label=label_distance
                )
    return state.state
