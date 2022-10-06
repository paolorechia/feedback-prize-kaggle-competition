from utils import MCRMSECalculator

import pandas as pd
cohesion_df = pd.read_csv("cohesion_predictions.csv")

mcrmse_calculator = MCRMSECalculator()
mcrmse_calculator.compute_column(cohesion_df["cohesion"], cohesion_df["cohesion_predictions"])
print(mcrmse_calculator.get_score())
