"""
PyTorch Dataset wrappers for UrbanSound8K.
Used by both CUDA (v2) and Colab (v3) versions.
"""
import os
import torch
from torch.utils.data import Dataset
from core.dataset import (
    load_audio, extract_mel_spectrogram,
    get_audio_path, SAMPLE_RATE
)


class MelSpectrogramDataset(Dataset):
    """
    On-the-fly mel spectrogram generation.
    No pre-generation — avoids storage bloat and lets us apply
    augmentation per-sample at training time.
    """
    def __init__(self, df, dataset_root, augment=False, mel_length=128):
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.augment = augment
        self.mel_length = mel_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = get_audio_path(self.dataset_root, row["fold"], row["slice_file_name"])
        audio = load_audio(path)

        if self.augment:
            audio = self._augment(audio)

        mel = extract_mel_spectrogram(audio, fixed_length=self.mel_length)
        # Shape: (1, n_mels, mel_length) — single channel like grayscale image
        mel = torch.from_numpy(mel).unsqueeze(0)
        label = torch.tensor(int(row["classID"]), dtype=torch.long)
        return mel, label

    def _augment(self, y):
        """Simple time-domain augmentations. Keep it cheap."""
        import numpy as np
        # Time shift: shift up to 10% of clip
        shift = int(np.random.uniform(-0.1, 0.1) * len(y))
        y = np.roll(y, shift)
        # Add small Gaussian noise
        y = y + np.random.normal(0, 0.005, y.shape).astype(y.dtype)
        return y
