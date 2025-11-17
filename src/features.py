import numpy as np
import pandas as pd
from scipy.signal import welch
import os

def bandpower(x, sf, band):
    fmin, fmax = band
    freqs, psd = welch(x, sf, nperseg=sf*2)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def extract_features(df, sf=100):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30)
    }
    rows = []
    for _, row in df.iterrows():
        x = row["signal"]
        features = {"subject": row["subject"], "label": row["label"]}
        total_power = bandpower(x, sf, (0.5, 30))
        for name, band in bands.items():
            features[f"{name}_rel_power"] = bandpower(x, sf, band) / total_power
        rows.append(features)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.read_pickle("data/processed/epochs.pkl")
    df_feat = extract_features(df)
    df_feat.to_csv("data/features.csv", index=False)
    print("✅ Saved data/features.csv")
