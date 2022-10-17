from typing import List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from model_catalog import Model


@dataclass
class TrainingContext:
    model: SentenceTransformer
    model_info: Model

    # Parameters
    warmup_steps: int
    weight_decay: float
    train_steps: int
    checkpoint_steps: int
    learning_rate: float
    num_epochs: int
    training_batch_size: int
    evaluation_batch_size: int

    attention_dropout: float
    hidden_dropout: float
    classifier_dropout: float

    is_multitask: bool
    attributes: List[str]
    attribute: str

    unique_id: str
    text_label: str
    test_size: float
    input_dataset: str
    max_samples_per_class: int
    use_evaluator: bool

    checkout_dir: str
    output_dir: str

    save_results_to_mongo: bool
    debug: bool


# TODO: use dataclass in the code
@dataclass
class Experiment:
    experiment_name: str
    base_model: str
    head_model: str
    num_iters: int
    batch_size: int
    loss_function: str
    test_size: float
    train_score: float
    test_score: float
    metric: str
    epochs: int
    unique_id: str
    learning_rate: float
    test_size: float
    attribute: str
    use_binary: bool
    use_sentences: bool
    setfit_model_max_length: int
    minimum_chunk_length: int
    attention_probs_dropout_prob: float
    hidden_dropout_prob: float
