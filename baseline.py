import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


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

# pyplot.show()
for attr in attributes:
    df_train[f"{attr}_label"] = df_train.apply(
        lambda x: labels[str(getattr(x, attr))], axis=1
    )
    print("Attribute: ", attr, " unique values: ", df_train[f"{attr}_label"].value_counts())

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

    # Train one classifier for each attribute
    label_dfs = {}
    for attribute in attributes:
        label_dfs[attribute] = {}
        y = df_train[f"{attribute}_label"]

        label_dfs[attribute]["train"] = y.filter(items=train_index, axis=0)
        label_dfs[attribute]["test"] = y.filter(items=test_index, axis=0)

        y_train = label_dfs[attribute]["train"]
        y_test = label_dfs[attribute]["test"]

        pipeline = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tdidf", TfidfTransformer()),
                (
                    "clf",
                    RandomForestClassifier(
                        max_depth=5, n_estimators=20, max_features=1, random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)

    idx = test_index[0]
    for x_ in X_test:
        column_wise_sum = 0
        for attribute in attributes:
            y_test_ = label_dfs[attribute]["test"]
            y_ = y_test_[idx]

            column_wise_sum += (
                reverse_labels[pipeline.predict([x_])[0]] - reverse_labels[y_]
            ) ** 2
        column_wise_sum = column_wise_sum / len(attributes)

        fold_MCRMSE += column_wise_sum
        idx += 1

    fold_MCRMSE = fold_MCRMSE / len(X_test)

    print("fold_MCRMSE: ", fold_MCRMSE)

    best_classifier = trained_classifiers[attribute]
    best_error = best_fold[attribute]

    if fold_MCRMSE < best_error:
        for attribute in attributes:
            best_fold[attribute] = fold_MCRMSE
            trained_classifiers[attribute] = pipeline


print(f"Best training MCRMSE: ", best_fold[attributes[0]])

# Predict on challenge set
challenge_X = df_challenge.full_text
print(challenge_X.head())
for attribute in attributes:
    predictions = trained_classifiers[attribute].predict(challenge_X)
    df_challenge[attribute] = [reverse_labels[pred] for pred in predictions]

df_challenge.drop("full_text", axis=1)
print(df_challenge.head())
df_challenge.to_csv("submission.csv", index=False)
