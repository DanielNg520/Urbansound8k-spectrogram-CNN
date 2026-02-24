"""
Core dataset utilities — shared across all 3 versions.
UrbanSound8K: 8732 clips, 10 classes, 10-fold cross-validation.
"""
import os
import numpy as np
import pandas as pd
import librosa

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CLASSES = [
    "air_conditioner", "car_horn", "children_playing",
    "dog_bark", "drilling", "engine_idling",
    "gun_shot", "jackhammer", "siren", "street_music"
]
N_CLASSES = 10
SAMPLE_RATE = 22050       # Resample everything to this
CLIP_DURATION = 4.0       # Seconds (pad/trim to this)
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000


# ──────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────
def load_audio(path, sr=SAMPLE_RATE, duration=CLIP_DURATION):
    """Load audio, resample, pad/trim to fixed length."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return np.zeros(int(sr * duration), dtype=np.float32)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Returns mean + std of MFCCs → 1D vector of size 2*n_mfcc.
    This is the classical ML baseline feature.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, fixed_length=128):
    """
    Returns log-mel spectrogram as 2D array (n_mels, fixed_length).
    Used as CNN input — treated like a grayscale image.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or trim time axis to fixed_length
    if log_mel.shape[1] < fixed_length:
        pad = fixed_length - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant',
                         constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :fixed_length]

    # Normalize to [0, 1]
    lo, hi = log_mel.min(), log_mel.max()
    if hi - lo > 1e-6:
        log_mel = (log_mel - lo) / (hi - lo)
    return log_mel.astype(np.float32)


# ──────────────────────────────────────────────
# Metadata loading
# ──────────────────────────────────────────────
def load_metadata(dataset_root):
    """
    Loads UrbanSound8K metadata CSV.
    Returns DataFrame with columns: slice_file_name, fold, classID, class
    """
    csv_path = os.path.join(dataset_root, "metadata", "UrbanSound8K.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    return pd.read_csv(csv_path)


def get_audio_path(dataset_root, fold, filename):
    return os.path.join(dataset_root, "audio", f"fold{fold}", filename)


# ──────────────────────────────────────────────
# 10-fold cross-validation splits
# ──────────────────────────────────────────────
def get_fold_splits(metadata):
    """
    Yields (train_df, test_df) for each of the 10 official folds.
    NEVER shuffle folds — the UrbanSound8K paper mandates this split.
    """
    folds = sorted(metadata["fold"].unique())
    for test_fold in folds:
        test_df = metadata[metadata["fold"] == test_fold].copy()
        train_df = metadata[metadata["fold"] != test_fold].copy()
        yield test_fold, train_df, test_df


# ──────────────────────────────────────────────
# Bulk feature pre-computation (optional, saves time)
# ──────────────────────────────────────────────
def precompute_mfcc_features(metadata, dataset_root, cache_path=None):
    """
    Pre-compute all MFCC features. Returns (X, y) arrays.
    Optionally saves/loads from cache_path (.npz).
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"[INFO] Loaded MFCC cache from {cache_path}")
        return data["X"], data["y"], data["folds"]

    X, y, folds = [], [], []
    for i, row in metadata.iterrows():
        path = get_audio_path(dataset_root, row["fold"], row["slice_file_name"])
        audio = load_audio(path)
        feat = extract_mfcc(audio)
        X.append(feat)
        y.append(row["classID"])
        folds.append(row["fold"])
        if (i + 1) % 500 == 0:
            print(f"  MFCC: {i+1}/{len(metadata)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    folds = np.array(folds, dtype=np.int64)

    if cache_path:
        np.savez(cache_path, X=X, y=y, folds=folds)
        print(f"[INFO] Saved MFCC cache to {cache_path}")

    return X, y, folds
