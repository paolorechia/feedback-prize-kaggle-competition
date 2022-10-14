"""Uses SentenceTransformer library directly instead."""
from dataclasses import dataclass
from typing import List, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation

from torch.utils.data import DataLoader
from load_data import create_attribute_stratified_split


@dataclass
class EvaluationDataset:
    sentences1: List[str]
    sentences2: List[str]
    scores: List[float]


@dataclass
class TrainingDataset:
    training_pairs: List[InputExample]


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


train_df, test_df = create_attribute_stratified_split(
    "cohesion", 0.8, dataset="sampled"
)
model_truncate_length = 512

# Let's see how it looks like :)
training_dataset: TrainingDataset = create_continuous_sentence_pairs(
    train_df, "full_text", "cohesion", model_truncate_length, "training"
)

for example in training_dataset.training_pairs[0:2]:
    print("Text 1: \n", example.texts[0])
    print("Text 2: \n", example.texts[1])
    print("Similarity Label:", example.label)

evaluation_dataset: EvaluationDataset = create_continuous_sentence_pairs(
    test_df, "full_text", "cohesion", model_truncate_length, "evaluation"
)

print(
    "Text 1: \n",
    evaluation_dataset.sentences1[0],
    "Text 2: \n",
    evaluation_dataset.sentences2[0],
    "Similarity Label:",
    evaluation_dataset.scores[0],
)
# Define the model. Either from scratch or by loading a pre-trained model
model_name = "all-distilroberta-v1"
model = SentenceTransformer(model_name)


evaluator = evaluation.EmbeddingSimilarityEvaluator(
    evaluation_dataset.sentences1,
    evaluation_dataset.sentences2,
    evaluation_dataset.scores,
    show_progress_bar=True,
    name="evaluator_output_{model_name}",
    write_csv=True,
)

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(
    training_dataset.training_pairs, shuffle=True, batch_size=16
)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    evaluator=evaluator,
    warmup_steps=100,
    output_path=f"st_output/{model_name}",
    save_best_model=True,
)
