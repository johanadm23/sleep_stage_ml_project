import pandas as pd
import numpy as np
import pickle
import os

def extract_features(df):
    feats = []
    for row in df.itertuples():
        x = row.signal
        feats.append({
            "subject": row.subject,
            "mean": np.mean(x),
            "std": np.std(x),
            "max": np.max(x),
            "min": np.min(x),
            "ptp": np.ptp(x),  # peak-to-peak amplitude
            "label": row.label
        })
    return pd.DataFrame(feats)

if __name__ == "__main__":
    print("Loading epochs...")
    df = pd.read_pickle("data/processed/epochs.pkl")

    print("Extracting features...")
    feats = extract_features(df)

    os.makedirs("data/features", exist_ok=True)
    feats.to_csv("data/features/features.csv", index=False)

    print("Saved data/features/features.csv")

