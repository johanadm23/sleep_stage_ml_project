import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy import signal, stats

def extract_time_domain_features(x):
    """Extract time-domain features from signal."""
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "var": np.var(x),
        "max": np.max(x),
        "min": np.min(x),
        "ptp": np.ptp(x),  # peak-to-peak
        "skewness": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "rms": np.sqrt(np.mean(x**2)),  # root mean square
        "zero_crossings": np.sum(np.diff(np.sign(x)) != 0),
    }

def extract_frequency_features(x, fs=100):
    """Extract frequency-domain features (EEG bands)."""
    # Compute power spectral density
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    
    # Define EEG frequency bands (Hz)
    bands = {
        'delta': (0.5, 4),    # Deep sleep
        'theta': (4, 8),      # Drowsiness
        'alpha': (8, 13),     # Relaxed, awake
        'beta': (13, 30),     # Alert, active thinking
        'gamma': (30, 50),    # High-level information processing
    }
    
    features = {}
    total_power = np.sum(psd)
    
    for band_name, (low, high) in bands.items():
        # Find indices for this frequency band
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.sum(psd[idx])
        
        features[f'{band_name}_power'] = band_power
        features[f'{band_name}_ratio'] = band_power / total_power if total_power > 0 else 0
    
    # Total power
    features['total_power'] = total_power
    
    # Spectral edge frequency (95% of power)
    cumsum_psd = np.cumsum(psd)
    sef95_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
    features['spectral_edge_95'] = freqs[sef95_idx[0]] if len(sef95_idx) > 0 else 0
    
    return features

def extract_statistical_features(x):
    """Extract additional statistical features."""
    # Percentiles
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    
    return {
        'q25': p25,
        'q50': p50,  # median
        'q75': p75,
        'iqr': p75 - p25,  # interquartile range
        'mad': np.median(np.abs(x - np.median(x))),  # median absolute deviation
    }

def extract_features(df, include_frequency=True):
    """Extract comprehensive features from signal data."""
    print(f"Extracting features from {len(df)} epochs...")
    
    feats = []
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing epochs"):
        x = row.signal
        
        # Combine all features
        features = {
            "subject": row.subject,
            "label": row.label
        }
        
        # Time domain features
        features.update(extract_time_domain_features(x))
        
        # Statistical features
        features.update(extract_statistical_features(x))
        
        # Frequency domain features (EEG-specific)
        if include_frequency:
            features.update(extract_frequency_features(x, fs=100))
        
        feats.append(features)
    
    return pd.DataFrame(feats)

if __name__ == "__main__":
    print("Loading epochs...")
    df = pd.read_pickle("data/processed/epochs.pkl")
    
    print(f"Loaded {len(df)} epochs from {df['subject'].nunique()} subjects")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Extract features
    print("\nExtracting features...")
    feats = extract_features(df, include_frequency=True)
    
    # Save features
    os.makedirs("data/features", exist_ok=True)
    feats.to_csv("data/features/features.csv", index=False)
    print(f"\n✅ Saved: data/features/features.csv")
    print(f"   Shape: {feats.shape}")
    print(f"   Features: {feats.shape[1] - 2} (excluding subject and label)")
    
    # Display feature names
    feature_cols = [col for col in feats.columns if col not in ['subject', 'label']]
    print(f"\nExtracted features ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    # Show sample
    print(f"\nSample features (first row):")
    print(feats.head(1).T)
    
    # Save feature metadata
    import json
    metadata = {
        'n_epochs': len(feats),
        'n_subjects': feats['subject'].nunique(),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'label_distribution': feats['label'].value_counts().to_dict()
    }
    
    with open("data/features/features_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved: data/features/features_metadata.json")

