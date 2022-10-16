"""Uses SentenceTransformer library directly instead."""

import os
from itertools import product
from uuid import uuid4

from experiment_schemas import TrainingContext
from model_catalog import ModelCatalog
from model_loader import load_model_with_dropout
from st_trainer import auto_trainer
from utils import attributes

# Static parameters
checkout_dir = "/data/feedback-prize/st-checkpoints/"
output_dir = "/data/feedback-prize/st-output/"
assert os.path.exists(checkout_dir)
assert os.path.exists(output_dir)
text_label = "full_text"
input_dataset = "full"
test_dataset = "full"
attribute = "cohesion"  # Not really used in multitask
is_multi_task = True  # We're only grid searching with multitask
use_evaluator = True

# Dynamic parameters
warmup_steps = [10]
num_epochs = [5]
train_steps = [50]
max_samples_per_class = [8]
learning_rate = [2e-5]
model_info = [ModelCatalog.DebertaV3]

test_size = [0.3]
# test_size = [0.3, 0.5, 0.7]
weight_decay = [0.01]
# weight_decay = [0.01, 0.05, 0.1]

attention_dropout = [0.0, 0.1, 0.5, 0.9]
hidden_dropout = [0.0, 0.1, 0.5, 0.9]
classifier_dropout = [0.0]

# Generate all combinations of parameters
params = list(
    product(
        warmup_steps,
        num_epochs,
        train_steps,
        max_samples_per_class,
        learning_rate,
        model_info,
        test_size,
        weight_decay,
        attention_dropout,
        hidden_dropout,
        classifier_dropout,
    )
)

print("Total number of experiments:", len(params))
for combination in params:
    (
        warmup_steps,
        num_epochs,
        train_steps,
        max_samples_per_class,
        learning_rate,
        model_info,
        test_size,
        weight_decay,
        attention_dropout,
        hidden_dropout,
        classifier_dropout,
    ) = combination

    model_name = model_info.model_name
    model_truncate_length = model_info.model_truncate_length
    batch_size = model_info.recommended_batch_size

    checkpoint_steps = train_steps
    unique_id = str(uuid4())

    # Load model with dropout :)
    model = load_model_with_dropout(
        model_info,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        classifier_dropout=classifier_dropout,
    )
    # Go!
    auto_trainer(
        TrainingContext(
            model=model,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            classifier_dropout=classifier_dropout,
            model_info=model_info,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            train_steps=train_steps,
            checkpoint_steps=checkpoint_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            is_multitask=is_multi_task,
            attributes=attributes,
            attribute=attribute,
            unique_id=unique_id,
            text_label=text_label,
            test_size=test_size,
            input_dataset=input_dataset,
            max_samples_per_class=max_samples_per_class,
            use_evaluator=use_evaluator,
            checkout_dir=checkout_dir,
            output_dir=output_dir,
        )
    )
    # Clear the model from memory
    del model
