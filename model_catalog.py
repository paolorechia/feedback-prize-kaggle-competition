from dataclasses import dataclass


@dataclass
class Model:
    model_name: str
    model_truncate_length: int
    recommended_batch_size: int
    is_from_library: bool


class ModelCatalog:
    AllMiniLML6v2 = Model(
        model_name="all-MiniLM-L6-v2",
        model_truncate_length=256,
        recommended_batch_size=512,
        is_from_library=True,
    )

    AllMpnetBasev2 = Model(
        model_name="all-mpnet-base-v2",
        model_truncate_length=384,
        recommended_batch_size=32,
        is_from_library=True,
    )

    AllMpnetBasev1 = Model(
        model_name="all-mpnet-base-v1",
        model_truncate_length=512,
        recommended_batch_size=32,
        is_from_library=True,
    )

    AllDistilrobertaV1 = Model(
        model_name="all-distilroberta-v1",
        model_truncate_length=512,
        recommended_batch_size=128,
        is_from_library=True,
    )

    BertBaseUncased = Model(
        model_name="bert-base-uncased",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3 = Model(
        model_name="microsoft/deberta-v3-base",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )


    DebertaV3Large = Model(
        model_name="microsoft/deberta-v3-large",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )