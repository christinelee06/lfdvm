import pandas as pd

df = pd.read_csv("metadata_best_model_sample.csv")

buffalo_df = df[df["best_model"] == "none"]
print(buffalo_df.describe())  # Summary statistics
