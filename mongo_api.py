import requests


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

    def register_score(self, experiment_name, train_score, test_score):
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
