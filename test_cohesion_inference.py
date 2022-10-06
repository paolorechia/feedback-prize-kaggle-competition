from setfit import SetFitModel

import os
import pandas as pd
from datetime import datetime
from utils import reverse_labels

output_dir = "./small_sets/"
data_dir = "/data/feedback-prize/"
train_filepath = os.path.join(data_dir, "train.csv")
challenge_df_filepath = os.path.join(data_dir, "test.csv")


model = SetFitModel.from_pretrained("./setfits")

train_df = pd.read_csv(train_filepath)

print("Predicting full dataset...")
t0 = datetime.now()
train_df["cohesion_predictions"] = model.predict(train_df["full_text"].tolist())
train_df["cohesion_predictions"] = train_df["cohesion_predictions"].apply(
    lambda x: reverse_labels[x]
)
t1 = datetime.now()
print("Elapsed time to run inference on full dataset: ", t1 - t0)

train_df[["text_id", "cohesion", "cohesion_predictions"]].to_csv(
    "cohesion_predictions.csv", index=False
)
print(train_df.head())
