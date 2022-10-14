"""Uses SentenceTransformer library directly instead."""

import os
from uuid import uuid4

from sentence_transformers import SentenceTransformer, losses, models

from model_catalog import ModelCatalog
from st_trainer import TrainingContext, auto_trainer
from utils import attributes

# model_info = ModelCatalog.AllMiniLML6v2
# model_info = ModelCatalog.AllDistilrobertaV1
# model_info = ModelCatalog.AllMpnetBasev1
# model_info = ModelCatalog.BertBaseUncased
model_info = ModelCatalog.DebertaV3
# model_info = ModelCatalog.DebertaV3Large

model_name = model_info.model_name
model_truncate_length = model_info.model_truncate_length
batch_size = model_info.recommended_batch_size


text_label = "full_text"
input_dataset = "full"
test_dataset = "full"
attribute = "cohesion"
is_multi_task = True

test_size = 0.5
max_samples_per_class = 8

use_evaluator = True
evaluator = None

unique_id = str(uuid4())

learning_rate = 2e-5
num_epochs = 2
checkpoint_steps = 50
weight_decay = 0.01
warmup_steps = 10
train_steps = 50

# Define the model. Either from scratch or by loading a pre-trained model
if model_info.is_from_library:
    model = SentenceTransformer(model_info.model_name)
else:
    word_embedding_model = models.Transformer(
        model_info.model_name, max_seq_length=model_info.model_truncate_length
    )
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

checkout_dir = "/data/feedback-prize/st-checkpoints/"
output_dir = "/data/feedback-prize/st-output/"
assert os.path.exists(checkout_dir)


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
