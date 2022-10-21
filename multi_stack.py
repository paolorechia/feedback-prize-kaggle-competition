import logging
import sys
import warnings
from warnings import warn

import optuna
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    LassoCV,
    OrthogonalMatchingPursuitCV,
    RidgeCV,
    SGDRegressor,
)
from sklearn.svm import SVR

from benchmark_models import benchmark_multi_stack
from load_data import create_train_test_df
from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import MultiEncodingStack
from utils import attributes

warnings.filterwarnings("ignore")

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# ModelBenchmark(rmse_score=0.4511073552381755, model_name='microsoft/deberta-v3-base', time_to_encode_in_seconds=530.214186)
# stack = ModelStack([ModelCatalog.DebertaV3, ModelCatalog.T5V1Base, ModelCatalog.T03B])

# Test_size is not tunable, because of the caching in the MultiHead class
# Before changing this, remember to implement to pass the test_size to the MultiHead class
train_df, test_df = create_train_test_df(test_size=0.2, dataset="full")

available_head_regressors = {
    "AdaBoostRegressor": AdaBoostRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "BayesianRidge": BayesianRidge,
    "ElasticNet": ElasticNet,
    "LassoCV": LassoCV,
    "OrthogonalMatchingPursuitCV": OrthogonalMatchingPursuitCV,
    "RidgeCV": RidgeCV,
    "SGDRegressor": SGDRegressor,
    "SVR": SVR,
}
regressors = [
    # "BayesianRidge",
    # "ElasticNet",
    # "OrthogonalMatchingPursuitCV",
    # "SGDRegressor",
    "RidgeCV",
    # "LassoCV",
    # "SVR",
    # "AdaBoostRegressor",
    # "GradientBoostingRegressor",
    # "RandomForestRegressor",
]

networks = [
    # "AllMiniLML6v2",
    # "AllMpnetBasev2",
    # "AllMpnetBasev1",
    # "AllDistilrobertaV1",
    # "RobertaLarge",
    # "BertBaseUncased",
    "DebertaV3",
    "DebertaV3Large",
    # "DebertaV3Small",
    # "DebertaV3XSmall",
    # "BartBase",
    # "BartLarge",
    # "AlbertV2",
    # "T5Base",
    # "T5Large",
    # "T5V1Base",
    # "T5V1Large",
    # "T03B",
    # "WordEmbeddingsKomminos",
    # "WordEmbeddingsGlove",
]


def objective(trial):
    # Integer parameter

    multi_stack = MultiEncodingStack()
    for attribute in attributes:
        num_stacks = trial.suggest_int(f"{attribute}_num_stacks", 1, 1)

        model_stack = []
        stack_trials = []
        for i in range(num_stacks):
            # Categorical parameter
            stack_trial = trial.suggest_categorical(f"{attribute}_stack_{i}", networks)
            stack_trials.append(stack_trial)

        for stack_trial in stack_trials:
            model_stack.append(ModelCatalog.from_string(stack_trial))

        head_regressor_trial = trial.suggest_categorical(
            f"{attribute}_head_regressor", regressors
        )
        head_regressor = available_head_regressors[head_regressor_trial]

        stack = ModelStack(
            model_stack,
        )
        multi_stack.add_encoding_stack(attribute, stack)
        multi_stack.add_head(attribute, head_regressor)

    result = benchmark_multi_stack(multi_stack, train_df, test_df)
    return result


study_name = "multi-stack-deberta-fever"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    direction="minimize",  # we want to minimize the error :)
)

study.optimize(objective, n_trials=10, n_jobs=1, show_progress_bar=True)
print(study.best_trial)
