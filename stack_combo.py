from benchmark_models import benchmark_stack
from load_data import create_train_test_df
from model_catalog import ModelCatalog
from model_stacker import ModelStack


# ModelBenchmark(rmse_score=0.4511073552381755, model_name='microsoft/deberta-v3-base', time_to_encode_in_seconds=530.214186)
# stack = ModelStack([ModelCatalog.DebertaV3, ModelCatalog.T5V1Base, ModelCatalog.T03B])

stack = ModelStack([ModelCatalog.DebertaV3, ModelCatalog.T5V1Base, ModelCatalog.T03B])

train_df, test_df = create_train_test_df(test_size=0.2, dataset="full")
result = benchmark_stack(stack, train_df, test_df)
print(result)
