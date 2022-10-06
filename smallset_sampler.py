import os
import pandas as pd
from pprint import pprint

output_dir = "./small_sets/"
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


label_sets = {
    k: {} for k in attributes
}
label_lengths = {
    k: {} for k in attributes
}
attribute_minimum_lengths = {
    k: 0 for k in attributes
}

for attr in attributes:
    for label in reverse_labels.keys():
        full_label = f"{attr}_label"

        label_set = df_train[df_train[full_label] == label]
        label_sets[attr][label] = label_set
        label_lengths[attr][label] = len(label_set)

for key, item in label_lengths.items():
    lenghts = list(item.values())
    minimum = min(lenghts)
    attribute_minimum_lengths[key] = minimum

pprint(attribute_minimum_lengths)

# Sample and build small sets
sampled_sets = {
    k: {} for k in attributes
}
full_sampled_set = pd.DataFrame()
for attr, item in label_lengths.items():    
    sampled_df = pd.DataFrame()
    for label, length in item.items():
        label_sample = label_sets[attr][label].sample(
            attribute_minimum_lengths[attr]
        )
        sampled_df = pd.concat([sampled_df, label_sample])
    sampled_sets[attr] = sampled_df
    full_sampled_set = pd.concat([full_sampled_set, sampled_df])


for key, item in sampled_sets.items():
    print(key, len(item))
    sampled_sets[key].to_csv(f"{output_dir}{key}.csv", index=False)

full_sampled_set.to_csv(f"{output_dir}full_sampled_set.csv", index=False)


# print sampled sets values counts to double check results 
for attr, sampled_set in sampled_sets.items():
    print("attr ", sampled_set[attr].value_counts())
