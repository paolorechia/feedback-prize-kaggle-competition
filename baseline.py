import os
import pandas as pd
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


data_dir = "/data/feedback-prize/"
train_filepath = os.path.join(data_dir, "train.csv")

df_train = pd.read_csv(train_filepath)
df_train["data_type"] = "Train"


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

for attribute in attributes:
    print("Checking attribute: %s" % attribute)

    df_train["label"] = df_train.apply(
        lambda x: labels[str(getattr(x, attribute))], axis=1
    )
    # pyplot.show()

    y = df_train.label
    X = df_train.full_text
    n_splits = 5
    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    i = 0

    # Mean columnwise root mean squared error
    # For now we'll compute this only for cohesion
    for train_index, test_index in kfold.split(X):
        MCRMSE = 0
        print("Fold: ", i)
        i += 1
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
                # (
                #     "clf",
                #     SVC(kernel="linear", C=0.025, probability=True, random_state=42),
                # ),
            ]
        )
        X_train = X.filter(items=train_index, axis=0)
        y_train = y.filter(items=train_index, axis=0)
        X_test = X.filter(items=test_index, axis=0)
        y_test = y.filter(items=test_index, axis=0)
        pipeline.fit(X_train, y_train)

        for _, (x_, y_) in enumerate(zip(X_test, y_test)):
            # print("Predicted: ", pipeline.predict([x])[0], " Actual: ", y)
            #     # print("")
            MCRMSE += (
                reverse_labels[pipeline.predict([x_])[0]] - reverse_labels[y_]
            ) ** 2
        MCRMSE = (MCRMSE / len(X_test)) ** 0.5
        print("MCRMSE: ", MCRMSE)
        # proba = pipeline.predict(X_test)
