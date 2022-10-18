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

    RobertaLarge = Model(
        model_name = "all-roberta-large-v1",
        model_truncate_length = 256,
        recommended_batch_size = 8,
        is_from_library = True,
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
        recommended_batch_size=1,  # Cannot run this model apparently due to memory constraints
        is_from_library=False,
    )

    DebertaV3Small = Model(
        model_name="microsoft/deberta-v3-small",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3XSmall = Model(
        model_name="microsoft/deberta-v3-xsmall",
        model_truncate_length=256,
        recommended_batch_size=64,
        is_from_library=False,
    )

    BartBase = Model(
        model_name="facebook/bart-base",
        model_truncate_length=1024,
        recommended_batch_size=8,
        is_from_library=False,
    )

    BartLarge = Model(
        model_name="facebook/bart-large-mnli",
        model_truncate_length=256,
        recommended_batch_size=1,
        is_from_library=False,
    )

    AlbertV2 = Model(
        model_name="albert-base-v2",
        model_truncate_length=128,
        recommended_batch_size=128,
        is_from_library=False,
    )

    T5Base = Model(
        model_name="t5-base",
        model_truncate_length=768,
        recommended_batch_size=8,
        is_from_library=False,
    )

    T5Large = Model(
        model_name="t5-large",
        model_truncate_length=512,
        recommended_batch_size=1,
        is_from_library=False,
    )

    T5V1Base = Model(
        model_name="google/t5-v1_1-base",
        model_truncate_length=512,
        recommended_batch_size=8,
        is_from_library=False,
    )

    T5V1Large = Model(
        model_name="google/t5-v1_1-large",
        model_truncate_length=512,
        recommended_batch_size=1,
        is_from_library=False,
    )

    T5LongGlobalBase = Model(
        model_name="google/long-t5-tglobal-base",
        model_truncate_length=4096,
        recommended_batch_size=8,
        is_from_library=False,
    ) 

    T03B = Model(
        model_name="bigscience/T0_3B",
        model_truncate_length=1024,
        recommended_batch_size=1,
        is_from_library=False,
    )

