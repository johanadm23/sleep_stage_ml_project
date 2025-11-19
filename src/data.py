import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

def parse_hypnogram(hyp_path):
    """Parse Sleep-EDF hypnogram file (.hyp format)."""
    annotations = []
    
    with open(hyp_path, 'rb') as f:
        data = f.read()
    
    epoch_sec = 30
    
    stage_map = {
        0: 'W', 1: '1', 2: '2', 3: '3', 4: '4', 5: 'R', 6: '?',
        ord('0'): 'W', ord('1'): '1', ord('2'): '2', ord('3'): '3',
        ord('4'): '4', ord('5'): 'R', ord('R'): 'R', ord('W'): 'W', ord('?'): '?',
    }
    
    for i, byte_val in enumerate(data):
        onset = i * epoch_sec
        duration = epoch_sec
        
        if byte_val in stage_map:
            stage = stage_map[byte_val]
        else:
            try:
                char = chr(byte_val)
                if char in ['0', '1', '2', '3', '4', '5', 'R', 'W', '?']:
                    stage = 'R' if char == '5' else 'W' if char == '0' else char
                else:
                    continue
            except:
                continue
        
        annotations.append({
            'onset': onset,
            'duration': duration,
            'description': f'Sleep stage {stage}'
        })
    
    onsets = [a['onset'] for a in annotations]
    durations = [a['duration'] for a in annotations]
    descriptions = [a['description'] for a in annotations]
    
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)

def load_subject(eeg_path, hypnogram_path, epoch_sec=30):
    """Load one subject's EEG and hypnogram data."""
    
    temp_path = None
    if eeg_path.endswith('.rec'):
        temp_path = eeg_path.replace('.rec', '_temp.edf')
        shutil.copy2(eeg_path, temp_path)
        eeg_path = temp_path
    
    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
        annot = parse_hypnogram(hypnogram_path)
        raw.set_annotations(annot, emit_warning=False)
        
        # Find target channel
        target_ch = None
        for ch in raw.ch_names:
            if "Fpz-Cz" in ch or "Pz-Oz" in ch:
                target_ch = ch
                break
        
        if target_ch:
            raw.pick([target_ch])
        else:
            raw.pick([raw.ch_names[0]])
        
        sfreq = raw.info["sfreq"]
        epoch_len = int(epoch_sec * sfreq)
        
        signals = []
        labels = []
        
        for ann in annot:
            desc = ann["description"]
            
            if "Sleep stage" not in desc:
                continue
            
            start = int(ann["onset"] * sfreq)
            duration_samples = int(ann["duration"] * sfreq)
            stop = start + min(duration_samples, epoch_len)
            
            if stop > raw.n_times:
                continue
            
            data, _ = raw[:, start:stop]
            
            if data.shape[1] < epoch_len:
                data = np.pad(data, ((0, 0), (0, epoch_len - data.shape[1])), mode='constant')
            
            signals.append(data.flatten()[:epoch_len])
            label = desc.replace("Sleep stage ", "").strip()
            labels.append(label)
        
        return np.array(signals), labels
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def build_dataframe(raw_dir, limit_subjects=None, remove_unknown=True):
    """Combine several subjects' data into a single DataFrame."""
    
    all_files = os.listdir(raw_dir)
    psg_files = []
    
    for f in all_files:
        if f.endswith('.hyp'):
            base_name = f.replace('.hyp', '')
            psg_candidates = [base_name + '.rec', base_name + '.edf', base_name]
            
            for candidate in psg_candidates:
                if candidate in all_files:
                    if candidate not in [p[0] for p in psg_files]:
                        psg_files.append((candidate, f))
                    break
    
    psg_files = sorted(psg_files, key=lambda x: x[0])
    
    if limit_subjects:
        psg_files = psg_files[:limit_subjects]
    
    print(f"\nFound {len(psg_files)} PSG-Hypnogram pairs")
    
    rows = []
    
    for psg_file, hyp_file in tqdm(psg_files, desc="Loading subjects"):
        subj = psg_file[:6]
        eeg_path = os.path.join(raw_dir, psg_file)
        hyp_path = os.path.join(raw_dir, hyp_file)
        
        try:
            signals, labels = load_subject(eeg_path, hyp_path)
            
            for sig, lab in zip(signals, labels):
                # Optionally skip unknown/movement epochs
                if remove_unknown and lab == '?':
                    continue
                
                rows.append({
                    "subject": subj,
                    "signal": sig,
                    "label": lab
                })
            
            print(f" {subj}: {len(signals)} epochs")
            
        except Exception as e:
            print(f" Error loading {psg_file}: {e}")
            continue
    
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["subject", "signal", "label"])
    
    return df

if __name__ == "__main__":
    raw_dir = "data/raw/sleep-edf-database-1.0.0"
    os.makedirs("data/processed", exist_ok=True)
    
    print("Loading Sleep-EDF database...")
    # Load ALL subjects
    df = build_dataframe(raw_dir, limit_subjects=None, remove_unknown=True)
    
    print(f"\n Results:")
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Subjects: {df['subject'].nunique()}")
    print(f"  Epochs per subject: {len(df) / df['subject'].nunique():.1f} (avg)")
    
    print(f"\n Label distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = 100 * count / len(df)
        print(f"  Stage {label}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\n Saving data...")
    df.to_pickle("data/processed/epochs.pkl")
    print("✅ Saved: data/processed/epochs.pkl")
    
    # Save summary statistics
    summary = {
        'total_epochs': len(df),
        'n_subjects': df['subject'].nunique(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'signal_shape': df['signal'].iloc[0].shape,
        'subjects': df['subject'].unique().tolist()
    }
    
    import json
    with open("data/processed/dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(" Saved: data/processed/dataset_summary.json")
