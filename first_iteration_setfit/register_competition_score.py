from mongo_api import MongoDataAPIClient

mongo_api = MongoDataAPIClient()

models_dir = "/data/feedback-prize/models/"
kaggle_datasets_dir = "/data/feedback-prize/kaggle-datasets"
models_to_deploy = [
    "cohesion_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "syntax_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "phraseology_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "vocabulary_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "grammar_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
    "conventions_head:SGDRegressor_iters:20_batchSize:128_lossFunction:CosineSimilarityLoss_testSize:0.8_id:d158_epoch_1",
]
competition_score = 0.54
for model_name in models_to_deploy:
    mongo_api.register_competition_score(model_name, competition_score)
