import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_subject(eeg_path, hypnogram_path, epoch_sec=30):
    """Load one subject's EEG and hypnogram data and return (signals, labels)."""
    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)

    # Keep Fpz-Cz channel only
    target_ch = [ch for ch in raw.ch_names if "Fpz-Cz" in ch]
    if not target_ch:
        target_ch = [raw.ch_names[0]]
    raw.pick_channels(target_ch)

    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    epoch_len = int(epoch_sec * raw.info["sfreq"])
    signals = []
    labels = []

    for ann in annotations:
        if "Sleep stage" not in ann["description"]:
            continue
        start = int(ann["onset"] * raw.info["sfreq"])
        stop = start + epoch_len
        if stop > len(raw.times):
            continue
        data, _ = raw[:, start:stop]
        signals.append(data.flatten())
        labels.append(ann["description"].replace("Sleep stage ", ""))

    return np.array(signals), labels

def build_dataframe(raw_dir, limit_subjects=3):
    """Combine several subjects' data into a single DataFrame."""
    eeg_files = sorted([f for f in os.listdir(raw_dir) if f.endswith("PSG.edf")])[:limit_subjects]
    rows = []

    for f in tqdm(eeg_files, desc="Loading subjects"):
        subj = f.split("-")[0]
        eeg_path = os.path.join(raw_dir, f)
        hyp_path = eeg_path.replace("PSG.edf", "Hypnogram.edf")
        if not os.path.exists(hyp_path):
            continue
        signals, labels = load_subject(eeg_path, hyp_path)
        for sig, lab in zip(signals, labels):
            rows.append({"subject": subj, "signal": sig, "label": lab})

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    raw_dir = "data/raw/sleep-cassette"
    os.makedirs("data/processed", exist_ok=True)
    df = build_dataframe(raw_dir)
    df.to_pickle("data/processed/epochs.pkl")
    print("✅ Saved data/processed/epochs.pkl")
