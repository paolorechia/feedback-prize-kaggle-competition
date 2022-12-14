from dataclasses import dataclass
from json import JSONEncoder
from operator import mod


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

import json


@dataclass
class ModelDescription:
    model_name: str
    model_truncate_length: int
    recommended_batch_size: int
    is_from_library: bool

    def __dict__(self):
        return {
            "model_name": str(self.model_name),
            "model_truncate_length": self.model_truncate_length,
            "recommended_batch_size": self.recommended_batch_size,
            "is_from_library": self.is_from_library,
        }

    def to_json(self):
        return json.dumps(self.__dict__())


class ModelCatalog:
    AllMiniLML6v2 = ModelDescription(
        model_name="all-MiniLM-L6-v2",
        model_truncate_length=256,
        recommended_batch_size=512,
        is_from_library=True,
    )

    AllMpnetBasev2 = ModelDescription(
        model_name="all-mpnet-base-v2",
        model_truncate_length=384,
        recommended_batch_size=32,
        is_from_library=True,
    )

    AllMpnetBasev1 = ModelDescription(
        model_name="all-mpnet-base-v1",
        model_truncate_length=512,
        recommended_batch_size=32,
        is_from_library=True,
    )

    AllDistilrobertaV1 = ModelDescription(
        model_name="all-distilroberta-v1",
        model_truncate_length=512,
        recommended_batch_size=128,
        is_from_library=True,
    )

    RobertaLarge = ModelDescription(
        model_name="all-roberta-large-v1",
        model_truncate_length=256,
        recommended_batch_size=8,
        is_from_library=True,
    )

    WordEmbeddingsKomminos = ModelDescription(
        model_name="average_word_embeddings_komninos",
        model_truncate_length=20000,
        recommended_batch_size=1024,
        is_from_library=True,
    )

    WordEmbeddingsGlove = ModelDescription(
        model_name="average_word_embeddings_glove.6B.300d",
        model_truncate_length=20000,
        recommended_batch_size=1024,
        is_from_library=True,
    )

    BertBaseUncased = ModelDescription(
        model_name="bert-base-uncased",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3 = ModelDescription(
        model_name="microsoft/deberta-v3-base",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3Large = ModelDescription(
        model_name="microsoft/deberta-v3-large",
        model_truncate_length=256,
        recommended_batch_size=1,  # Cannot run this model apparently due to memory constraints
        is_from_library=False,
    )

    DebertaV3Small = ModelDescription(
        model_name="microsoft/deberta-v3-small",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3XSmall = ModelDescription(
        model_name="microsoft/deberta-v3-xsmall",
        model_truncate_length=256,
        recommended_batch_size=64,
        is_from_library=False,
    )

    DebertaV3Squad2 = ModelDescription(
        model_name="deepset/deberta-v3-base-squad2",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3LargeMNLI = ModelDescription(
        model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3ABSA = ModelDescription(
        model_name="yangheng/deberta-v3-base-absa-v1.1",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3ArgumentativeWriting = ModelDescription(
        model_name="kitkeat/deberta-v3-base-argumentativewriting",
        model_truncate_length=256,
        recommended_batch_size=32,
        is_from_library=False,
    )

    DebertaV3MNLI = ModelDescription(
        model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3NLI = ModelDescription(
        model_name="cross-encoder/nli-deberta-v3-base",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3NDugar = ModelDescription(
        model_name="NDugar/debertav3-mnli-snli-anli",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3DebiasedNLI = ModelDescription(
        model_name="tomhosking/deberta-v3-base-debiased-nli",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3TwitterEmotion = ModelDescription(
        model_name="Emanuel/twitter-emotion-deberta-v3-base",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3GoEmotions = ModelDescription(
        model_name="mrm8488/deberta-v3-base-goemotions",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3SetFit = ModelDescription(
        model_name="SetFit/deberta-v3-base__sst2__all-train",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    DebertaV3Cola = ModelDescription(
        model_name="yevheniimaslov/deberta-v3-base-cola",
        model_truncate_length=256,
        recommended_batch_size=16,
        is_from_library=False,
    )

    BartBase = ModelDescription(
        model_name="facebook/bart-base",
        model_truncate_length=1024,
        recommended_batch_size=8,
        is_from_library=False,
    )

    BartLarge = ModelDescription(
        model_name="facebook/bart-large-mnli",
        model_truncate_length=256,
        recommended_batch_size=1,
        is_from_library=False,
    )

    AlbertV2 = ModelDescription(
        model_name="albert-base-v2",
        model_truncate_length=128,
        recommended_batch_size=128,
        is_from_library=False,
    )

    T5Base = ModelDescription(
        model_name="t5-base",
        model_truncate_length=768,
        recommended_batch_size=8,
        is_from_library=False,
    )

    T5Large = ModelDescription(
        model_name="t5-large",
        model_truncate_length=512,
        recommended_batch_size=1,
        is_from_library=False,
    )

    T5V1Base = ModelDescription(
        model_name="google/t5-v1_1-base",
        model_truncate_length=512,
        recommended_batch_size=8,
        is_from_library=False,
    )

    T5V1Large = ModelDescription(
        model_name="google/t5-v1_1-large",
        model_truncate_length=512,
        recommended_batch_size=1,
        is_from_library=False,
    )

    T5LongGlobalBase = ModelDescription(
        model_name="google/long-t5-tglobal-base",
        model_truncate_length=4096,
        recommended_batch_size=8,
        is_from_library=False,
    )

    T03B = ModelDescription(
        model_name="bigscience/T0_3B",
        model_truncate_length=1024,
        recommended_batch_size=1,
        is_from_library=False,
    )

    GPTNeo = ModelDescription(
        model_name="EleutherAI/gpt-neo-1.3B",
        model_truncate_length=2048,
        recommended_batch_size=1,
        is_from_library=False,
    )

    GPTNeo2 = ModelDescription(
        model_name="EleutherAI/gpt-neo-2.7B",
        model_truncate_length=2048,
        recommended_batch_size=1,
        is_from_library=False,
    )

    Pythia1 = ModelDescription(
        model_name="EleutherAI/pythia-1.3b",
        model_truncate_length=2048,
        recommended_batch_size=1,
        is_from_library=False,
    )

    TFIDF = ModelDescription(
        model_name="tfidf",
        model_truncate_length=20000,
        recommended_batch_size=1,
        is_from_library=False,
    )

    @staticmethod
    def from_string(model_catalog_name: str):
        return getattr(ModelCatalog, model_catalog_name)


if __name__ == "__main__":
    # Test that 'Model' is JSON serializable
    print(json.dumps(ModelCatalog.DebertaV3XSmall))
