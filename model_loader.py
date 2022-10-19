import torch
from warnings import warn
from model_catalog import ModelDescription
from sentence_transformers import SentenceTransformer, models
import os
import json

dropout_dir = "/data/dropout_test/"


def get_dropout_model_path(model_info: ModelDescription):
    return os.path.join(dropout_dir, model_info.model_name)


def load_model_with_dropout(
    model_info: ModelDescription,
    attention_dropout: float,
    hidden_dropout: float,
    classifier_dropout: float = 0.0,
) -> SentenceTransformer:
    dropout_path = get_dropout_model_path(model_info)
    if not os.path.exists(dropout_path):
        model = _load_model(model_info)
        model.save(dropout_path)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(dropout_path, "config.json"), "r") as fp:
        config = json.load(fp)

        if "attention_probs_dropout_prob" in config:
            config["attention_probs_dropout_prob"] = attention_dropout
        else:
            warn("Attention dropout not found in config")
        if "hidden_dropout_prob" in config:
            config["hidden_dropout_prob"] = hidden_dropout
        else:
            warn("Hidden dropout not found in config")
        if classifier_dropout > 0.0:
            if "classifier_dropout" in config:
                config["classifier_dropout"] = classifier_dropout
            else:
                warn("Classifier dropout not found in config")

    with open(os.path.join(dropout_path, "config.json"), "w") as fp:
        json.dump(config, fp, indent=4)

    model = _load_model_from_path(dropout_path)
    return model


def _load_model_from_path(path):
    model = SentenceTransformer(path)
    return model


def _load_model(model_info: ModelDescription):
    # Define the model. Either from scratch or by loading a pre-trained model
    if model_info.is_from_library:
        model = _instantiate_library_model(model_info)
    else:
        model = _instantiate_custom_model(model_info)
    return model


def _instantiate_library_model(model_info: ModelDescription):
    model = SentenceTransformer(model_info.model_name)
    return model


def _instantiate_custom_model(model_info: ModelDescription):
    word_embedding_model = models.Transformer(
        model_info.model_name,
        max_seq_length=model_info.model_truncate_length,
    )
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model
