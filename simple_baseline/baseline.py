"""
Baseline without balancing the data, RandomForestClassifier achieves a score of 0.63 (MCRMSE) (0.65 on Fold).
Applying class weights to the random forest model decreaes the score to 1.41 (lower is better) 
With RidgeClassifierCV, we can get a score of 0.62 on Fold

If we instead model it as a regression problem, we can get a score of 0.54 on Fold with RidgeCV
If we apply round_border_score function, makes no absolute difference.

"""

import os

import pandas as pd
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_rmse_score(y_true, y_pred):
    rmse_scores = []
    for i in range(len(attributes)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    return np.mean(rmse_scores)


def round_border_score(float_score: float):
    if float_score < 1.0:
        return 1.0
    if float_score > 5.0:
        return 5.0
    return float_score


output_dir = "./"
data_dir = "/data/feedback-prize/"
train_filepath = os.path.join(data_dir, "train.csv")
challenge_df_filepath = os.path.join(data_dir, "test.csv")


df_train = pd.read_csv(train_filepath)
df_challenge = pd.read_csv(challenge_df_filepath)


print(df_train.columns)
print(df_train.head())

attributes = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]


labels = {
    "1.0": "terrible",
    "1.5": "bad",
    "2.0": "poor",
    "2.5": "fair",
    "3.0": "average",
    "3.5": "good",
    "4.0": "great",
    "4.5": "excellent",
    "5.0": "perfect",
}
reverse_labels = {v: float(k) for k, v in labels.items()}

# for attr in attributes:
#     df_train[f"{attr}_label"] = df_train.apply(
#         lambda x: labels[str(getattr(x, attr))], axis=1
#     )
#     print(
#         "Attribute: ",
#         attr,
#         " unique values: ",
#         df_train[f"{attr}_label"].value_counts(),
#     )

X = df_train.full_text
n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
i = 0


# Mean columnwise root mean squared error
# For now we'll compute this only for cohesion
trained_classifiers = {k: None for k in attributes}
best_fold = {k: 1000.0 for k in attributes}

for train_index, test_index in kfold.split(X):
    fold_MCRMSE = 0

    print("Fold: ", i)
    i += 1

    X_train = X.filter(items=train_index, axis=0)
    X_test = X.filter(items=test_index, axis=0)

    predictions_df = pd.DataFrame()
    # Train one classifier for each attribute
    label_dfs = {}
    for attribute in attributes:
        label_dfs[attribute] = {}
        y = df_train[attribute]
        # y = df_train[f"{attribute}_label"]

        label_dfs[attribute]["train"] = y.filter(items=train_index, axis=0)
        label_dfs[attribute]["test"] = y.filter(items=test_index, axis=0)

        y_train = label_dfs[attribute]["train"]
        y_test = label_dfs[attribute]["test"]

        weights = {}
        for label in y_train.unique():
            # print(y_train.value_counts())
            weights[label] = 1 / y_train.value_counts()[label]

        pipeline = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tdidf", TfidfTransformer()),
                # ("clf", RidgeClassifierCV()),
                ("clf", RidgeCV()),
            ]
        )
        pipeline.fit(X_train, y_train)

        # predictions_df[attribute] = [
        #     reverse_labels[l] for l in pipeline.predict(X_test)
        # ]
        predictions_df[attribute] = [
            round_border_score(p) for p in pipeline.predict(X_test)
        ]

    test_df = df_train.filter(items=test_index, axis=0)
    fold_MCRMSE = calculate_rmse_score(
        predictions_df[attributes].values, test_df[attributes].values
    )

    print("MCRMSE: ", fold_MCRMSE)

    best_classifier = trained_classifiers[attribute]
    best_error = best_fold[attribute]

    if fold_MCRMSE < best_error:
        for attribute in attributes:
            best_fold[attribute] = fold_MCRMSE
            trained_classifiers[attribute] = pipeline


print(f"Best training MCRMSE: ", best_fold[attributes[0]])

# Predict on challenge set
# challenge_X = df_challenge.full_text
# print(challenge_X.head())
# for attribute in attributes:
#     predictions = trained_classifiers[attribute].predict(challenge_X)
#     df_challenge[attribute] = [reverse_labels[pred] for pred in predictions]

# df_challenge.drop(columns=["full_text"], inplace=True)
# print(df_challenge.head())
# output_path = os.path.join(output_dir, "submission.csv")
# df_challenge.to_csv(output_path, index=False, float_format="%.1f")
