import logging
import sys
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

from benchmark_models import benchmark_stack
from load_data import create_train_test_df
from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import (
    MultiClassMultiHeadSentenceTransformerModel,
)
from utils import attributes
import warnings

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
    "BayesianRidge",
    "ElasticNet",
    "OrthogonalMatchingPursuitCV",
    "SGDRegressor",
    "RidgeCV",
    "LassoCV",
    "SVR",
    "AdaBoostRegressor",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
]

networks = [
    # "AllMiniLML6v2",
    # "AllMpnetBasev2",
    # "AllMpnetBasev1",
    # "AllDistilrobertaV1",
    # "RobertaLarge",
    # "BertBaseUncased",
    # "DebertaV3",
    # "DebertaV3Large",
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
    "WordEmbeddingsKomninos",
    "WordEmbeddingsGlove",
]

def objective(trial):
    # Integer parameter
    num_stacks = trial.suggest_int("num_stacks", 1, 2)

    model_stack = []
    stack_trials = []
    for i in range(num_stacks):
        # Categorical parameter
        stack_trial = trial.suggest_categorical(f"stack_{i}", networks)
        stack_trials.append(stack_trial)

    for stack_trial in stack_trials:
        model_stack.append(ModelCatalog.from_string(stack_trial))

    heads = {}
    for attribute in attributes:
        head_regressor_trial = trial.suggest_categorical(
            f"{attribute}_head_regressor", regressors
        )
        head_regressor = available_head_regressors[head_regressor_trial]
        heads[attribute] = head_regressor

    stack = ModelStack(
        model_stack,
    )

    multi_class_multi_head = MultiClassMultiHeadSentenceTransformerModel(stack)
    for key, item in heads.items():
        print(item)
        if item == RidgeCV:
            alpha = trial.suggest_float(f"alpha_{key}_{item}", 0.1, 100.0)
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=False,
                head_model=item,
                alphas=[alpha],
            )
        elif item == LassoCV:
            n_alphas = trial.suggest_int(f"n_alphas_{key}_{item}", 10, 1000)
            max_iter = trial.suggest_int(f"max_iter_{key}_{item}", 1000, 5000)
            tol = trial.suggest_float(f"tol_{key}_{item}", 1e-6, 1e-3)
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=False,
                head_model=item,
                n_alphas=n_alphas,
                max_iter=max_iter,
                tol=tol,
                random_state=0,
            )
        elif item == AdaBoostRegressor:
            n_estimators = trial.suggest_int(f"n_estimators_{key}_{item}", 10, 100)
            learning_rate = trial.suggest_float(
                f"learning_rate_{key}_{item}", 0.1, 10.0
            )
            loss = trial.suggest_categorical(
                f"loss_{key}_{item}", ["linear", "square", "exponential"]
            )
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=False,
                head_model=item,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=0,
            )
        elif item == GradientBoostingRegressor:
            n_estimators = trial.suggest_int(f"n_estimators_{key}_{item}", 10, 100)
            learning_rate = trial.suggest_float(f"learning_rate_{key}_{item}", 0.1, 1.0)
            loss = trial.suggest_categorical(
                f"loss_{key}_{item}",
                ["squared_error", "absolute_error", "huber", "quantile"],
            )
            subsample = trial.suggest_float(f"subsample_{key}_{item}", 0.1, 1.0)
            criterion = trial.suggest_categorical(
                f"criterion_{key}_{item}", ["friedman_mse", "squared_error"]
            )
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=False,
                head_model=item,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                subsample=subsample,
                criterion=criterion,
                random_state=0,
            )
        elif item == RandomForestRegressor:
            n_estimators = trial.suggest_int(f"n_estimators_{key}_{item}", 10, 1000)
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=False,
                head_model=item,
                n_estimators=n_estimators,
                random_state=0,
            )
        elif item == BayesianRidge:
            alpha_1 = trial.suggest_float(f"alpha_1_{key}_{item}", 1e-6, 1e-3)
            alpha_2 = trial.suggest_float(f"alpha_2_{key}_{item}", 1e-6, 1e-3)
            lambda_1 = trial.suggest_float(f"lambda_1_{key}_{item}", 1e-6, 1e-3)
            lambda_2 = trial.suggest_float(f"lambda_2_{key}_{item}", 1e-6, 1e-3)
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=False,
                head_model=item,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
            )
        elif item == SVR:
            kernel = trial.suggest_categorical(f"kernel_{key}_{item}", ["poly", "rbf"])
            C = trial.suggest_float(f"C_{key}_{item}", 0.001, 100.0)
            degree = trial.suggest_int(f"degree_{key}_{item}", 3, 8)
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=True,
                head_model=item,
                kernel=kernel,
                C=C,
                degree=degree,
            )
        elif item == OrthogonalMatchingPursuitCV:
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=True,
                head_model=item,
                normalize=False,
            )
        elif item == ElasticNet:
            alpha = trial.suggest_float(f"alpha_{key}_{item}", 0.1, 100.0)
            l1_ratio = trial.suggest_float(f"l1_ratio_{key}_{item}", 0.1, 1.0)
            multi_class_multi_head.add_head(
                attribute=key,
                use_scaler=True,
                head_model=item,
                alpha=alpha,
                l1_ratio=l1_ratio,
            )
        else:
            warn(
                f"Head regressor {item} does not have parameter trials implemented yet."
            )
            multi_class_multi_head.add_head(
                attribute=key, use_scaler=True, head_model=item
            )

    result = benchmark_stack(stack, multi_class_multi_head, train_df, test_df)
    return result.rmse_score


study_name = (
    "multi-class-multi-head-embeddings-test"  # Unique identifier of the study.
)
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    direction="minimize",  # we want to minimize the error :)
)

study.optimize(objective, n_trials=100, n_jobs=8, show_progress_bar=True)
print(study.best_trial)
