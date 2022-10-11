import requests
from dataclasses import dataclass
import warnings

# TODO: use dataclass in the code
@dataclass
class Experiment:
    experiment_name: str
    base_model: str
    head_model: str
    num_iters: int
    batch_size: int
    loss_function: str
    test_size: float
    train_score: float
    test_score: float
    metric: str
    epochs: int
    unique_id: str
    learning_rate: float
    test_size: float
    attribute: str
    use_binary: bool
    use_sentences: bool
    setfit_model_max_length: int
    minimum_chunk_length: int


class MongoDataAPIClient:
    def __init__(self):
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
        self._collection = "setfit"
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
