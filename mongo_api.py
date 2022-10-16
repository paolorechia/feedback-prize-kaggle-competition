import warnings

import requests

from experiment_schemas import Experiment, TrainingContext


class MongoDataAPIClient:
    def __init__(self, collection="setfit"):
        with open(".mongo_api_key", "r") as fp:
            self._api_key = fp.read().strip()

        with open(".mongo_url", "r") as fp:
            self._mongo_url = fp.read().strip()

        self._session = requests.Session()
        self._headers = {
            "Content-Type": "application/json",
            "Access-Control-Request-Headers": "*",
            "api-key": self._api_key,
        }
        self._collection = collection
        self._database = "feedback-prize-kaggle-competition"
        self._data_source = "Cluster0"

    def _call_api(self, action, data):
        payload = {
            "collection": self._collection,
            "database": self._database,
            "dataSource": self._data_source,
        }
        payload.update(data)
        response = self._session.post(
            f"{self._mongo_url}/action/{action}", headers=self._headers, json=payload
        )
        print(response)
        if response.status_code not in [200, 201]:
            print(f"API call failed: {response.content.decode()}")
            return None
        return response.json()

    def register_st_training_context(self, training_context: TrainingContext):
        data = {
            "document": {
                "model_name": training_context.model_info.model_name,
                "warmup_steps": training_context.warmup_steps,
                "weight_decay": training_context.weight_decay,
                "learning_rate": training_context.learning_rate,
                "checkpoint_steps": training_context.checkpoint_steps,
                "num_epochs": training_context.num_epochs,
                "batch_size": training_context.batch_size,
                "is_multitask": training_context.is_multitask,
                "attribute": training_context.attribute,
                "unique_id": training_context.unique_id,
                "test_size": training_context.test_size,
                "input_dataset": training_context.input_dataset,
                "max_samples_per_class": training_context.max_samples_per_class,
                "use_evaluator": training_context.use_evaluator,
                "checkout_dir": training_context.checkout_dir,
                "output_dir": training_context.output_dir,
                "evaluation_scores": [],
                "mcrmse_scores": {
                    "all": [],
                    "cohesion": [],
                    "syntax": [],
                    "vocabulary": [],
                    "phraseology": [],
                    "grammar": [],
                    "conventions": [],
                },
            }
        }
        return self._call_api("insertOne", data)

    def append_training_context_scores(
        self, unique_id, evaluation_score, mcrmse_scores
    ):
        data = {
            "filter": {"unique_id": unique_id},
            "update": {
                "$push": {
                    "evaluation_scores": evaluation_score,
                    "mcrmse_scores.all": mcrmse_scores["all"],
                    "mcrmse_scores.cohesion": mcrmse_scores["cohesion"],
                    "mcrmse_scores.syntax": mcrmse_scores["syntax"],
                    "mcrmse_scores.vocabulary": mcrmse_scores["vocabulary"],
                    "mcrmse_scores.phraseology": mcrmse_scores["phraseology"],
                    "mcrmse_scores.grammar": mcrmse_scores["grammar"],
                    "mcrmse_scores.conventions": mcrmse_scores["conventions"],
                }
            },
        }
        return self._call_api("updateOne", data)

    def register_experiment(self, experiment: Experiment):
        data = {
            "document": {
                "experiment_name": experiment.experiment_name,
                "base_model": experiment.base_model,
                "head_model": experiment.head_model,
                "num_iters": experiment.num_iters,
                "batch_size": experiment.batch_size,
                "loss_function": experiment.loss_function,
                "test_size": experiment.test_size,
                "train_score": experiment.train_score,
                "test_score": experiment.test_score,
                "metric": experiment.metric,
                "epochs": experiment.epochs,
                "unique_id": experiment.unique_id,
                "learning_rate": experiment.learning_rate,
                "test_size": experiment.test_size,
                "attribute": experiment.attribute,
                "use_binary": experiment.use_binary,
                "use_sentences": experiment.use_sentences,
                "setfit_model_max_length": experiment.setfit_model_max_length,
                "minimum_chunk_length": experiment.minimum_chunk_length,
                "attention_probs_dropout_prob": experiment.attention_probs_dropout_prob,
                "hidden_dropout_prob": experiment.hidden_dropout_prob,
            }
        }
        return self._call_api("insertOne", data)

    def register_score(self, experiment_name, train_score, test_score):
        warnings.warn("Deprecation warning: Use register_experiment instead")
        data = {
            "document": {
                "experiment_name": experiment_name,
                "score": train_score,
                "test_score": test_score,
                "train_score": train_score,
            }
        }
        return self._call_api("insertOne", data)

    def register_competition_score(self, experiment_name, competition_score):
        data = {
            "filter": {
                "experiment_name": experiment_name,
            },
            "update": {"$set": {"competition_score": competition_score}},
        }
        return self._call_api("updateOne", data)

    def find_experiments(self, experiment_name):
        data = {"filter": {"experiment_name": experiment_name}}
        return self._call_api("find", data)
