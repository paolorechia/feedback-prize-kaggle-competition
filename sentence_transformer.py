from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

dataset = load_dataset(
    "csv",
    data_files={
        "train": "small_sets/cohesion.csv",
        "test": "small_sets/full_sampled_set.csv",
    },
)

dataset["train"] = dataset["train"].rename_column("cohesion_label", "label")
dataset["train"] = dataset["train"].rename_column("full_text", "text")

dataset["test"] = dataset["test"].rename_column("cohesion_label", "label")
dataset["test"] = dataset["test"].rename_column("full_text", "text")


train_ds = dataset["train"]
test_ds = dataset["test"]

print(train_ds)
print(test_ds)
# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_epochs=24,
    column_mapping={
        "text": "text",
        "label": "label",
    },
)
# Train!
trainer.train()
trainer.model._save_pretrained("./setfits")