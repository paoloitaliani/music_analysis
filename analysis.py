import pandas as pd

df = pd.read_csv("/Users/Niolo/Desktop/Github/music_analysis/outputs/full_dataset_features.csv")
mask = df["artist_name"] == "The Beatles"
masked = df[mask]

