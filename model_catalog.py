from dataclasses import dataclass


@dataclass
class Model:
    model_name: str
    model_truncate_length: int
    recommended_batch_size: int

class ModelCatalog:
    AllMiniLML6v2 = Model(
        model_name="all-MiniLM-L6-v2",
        model_truncate_length=256,
        recommended_batch_size=512,
    )
    AllMpnetBasev2 = Model(
        model_name="all-mpnet-base-v2",
        model_truncate_length=384,
        recommended_batch_size=32
    )
    AllMpnetBasev1 = Model(
        model_name="all-mpnet-base-v1",
        model_truncate_length=512,
        recommended_batch_size=32
    )
    AllDistilrobertaV1 = Model(
        model_name="all-distilroberta-v1",
        model_truncate_length=512,
        recommended_batch_size=128
    )
