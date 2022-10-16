"""Uses SentenceTransformer library directly instead."""

import os
from itertools import product
from uuid import uuid4

from sentence_transformers import SentenceTransformer, models

from experiment_schemas import TrainingContext
from model_catalog import ModelCatalog
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
model_info = [
    ModelCatalog.AllMiniLML6v2,
    ModelCatalog.AllDistilrobertaV1,
    ModelCatalog.AllMpnetBasev1,
    ModelCatalog.BertBaseUncased,
    ModelCatalog.DebertaV3,
    ModelCatalog.DebertaV3Large,
]
test_size = [0.3, 0.5, 0.7]
weight_decay = [0.01, 0.05, 0.1]

# Generate all combinations of parameters
params = product(
    warmup_steps,
    num_epochs,
    train_steps,
    max_samples_per_class,
    learning_rate,
    model_info,
    test_size,
    weight_decay,
)

print("Total number of experiments:", len(list(params)))
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
    ) = combination

    model_name = model_info.model_name
    model_truncate_length = model_info.model_truncate_length
    batch_size = model_info.recommended_batch_size

    checkpoint_steps = train_steps
    unique_id = str(uuid4())

    # Define the model. Either from scratch or by loading a pre-trained model
    if model_info.is_from_library:
        model = SentenceTransformer(model_info.model_name)
    else:
        word_embedding_model = models.Transformer(
            model_info.model_name,
            max_seq_length=model_info.model_truncate_length,
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Go!
    auto_trainer(
        TrainingContext(
            model=model,
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
