from setfit import SetFitModel

models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "all-mpnet-base-v1",
    "all-distilroberta-v1",
]
for model_ in models:
    # Save model config
    model = SetFitModel._from_pretrained(f"sentence-transformers/{model_}")
    model._save_pretrained(f"dropout_test/{model_}")
