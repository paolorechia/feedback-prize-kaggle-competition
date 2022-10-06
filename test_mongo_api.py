from mongo_api import MongoDataAPIClient

mongo_api = MongoDataAPIClient()

mongo_api.register_score("test", 0.5)

experiment = mongo_api.find_experiments("test")
print(experiment)