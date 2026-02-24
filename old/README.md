# ECE 176 — UrbanSound8K Classification Project

**MFCC + SVM vs. CNN on Mel Spectrograms**  
10-fold cross-validation | 3 runtime targets

---

## File Structure

```
urbansound/
├── core/
│   ├── dataset.py          # Audio loading, feature extraction, metadata, fold splits
│   ├── torch_dataset.py    # PyTorch Dataset wrapper (used by v2 & v3)
│   ├── models.py           # UrbanCNN (full) and LightCNN (CPU-friendly)
│   └── evaluation.py       # Metrics, print/save results
├── v1_cpu.py               # Version 1: CPU-only
├── v2_cuda.py              # Version 2: CUDA multi-session
├── v3_colab_pro.ipynb      # Version 3: Colab Pro notebook
├── requirements.txt
└── README.md
```

---

## Version 1: CPU (Intel cores)

No GPU needed. Uses `LightCNN` (~300K params) to keep training tractable on CPU.

```bash
pip install -r requirements.txt

# Full run (SVM baseline + CNN)
python v1_cpu.py --dataset_root /path/to/UrbanSound8K

# Options
python v1_cpu.py --dataset_root /path/to/UrbanSound8K \
    --epochs 20 \        # default 30; reduce if too slow
    --jobs 8 \           # CPU cores for SVM (-1 = all)
    --skip_cnn           # run only SVM baseline
    --skip_svm           # run only CNN
```

**Expected time on 8-core Intel:**
- MFCC pre-computation: ~10 min (cached after first run)
- SVM training: ~5-10 min per fold
- CNN training (30 epochs): ~20-40 min per fold → ~4-7 hours total

To speed up: reduce `--epochs 15` and call `--skip_svm` after baseline is done.

---

## Version 2: CUDA (PyTorch) — Multi-session

Uses `UrbanCNN` (~1.2M params) with AMP (automatic mixed precision).  
Saves progress per fold — kill and resume anytime.

```bash
# Install with CUDA support (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install librosa scikit-learn numpy pandas tqdm matplotlib seaborn

# First run
python v2_cuda.py --dataset_root /path/to/UrbanSound8K

# Resume after interruption
python v2_cuda.py --dataset_root /path/to/UrbanSound8K --resume

# Options
python v2_cuda.py --dataset_root /path/to/UrbanSound8K \
    --resume \
    --epochs 60 \
    --batch_size 128 \   # increase for >8GB VRAM
    --workers 4 \
    --skip_svm           # CNN only
```

**Checkpoint behavior:**
- `checkpoints/cuda/foldN_best.pt` — best weights for fold N
- `checkpoints/cuda/foldN_resume.pt` — mid-fold training state (deleted on fold completion)
- `checkpoints/cuda/progress.json` — which folds are done + per-fold accuracy

**Expected time on RTX 3080 (60 epochs):**
- SVM: ~5 min/fold
- CNN: ~3-5 min/fold → ~30-50 min total for CNN

---

## Version 3: Colab Pro

Open `v3_colab_pro.ipynb` in Colab. No local files needed — everything is self-contained.

**Hardware selection (Runtime → Change runtime type → Hardware accelerator):**

| Option | VRAM | Recommended batch size | Notes |
|--------|------|----------------------|-------|
| T4 GPU | 16 GB | 64 | Default Pro option |
| V100 GPU | 16 GB | 128 | Faster than T4 |
| A100 GPU | 40 GB | 256 | Fastest, Pro+ only |
| TPU | — | — | **Don't use** — PyTorch TPU setup is complex, not worth it here |

**Multi-session workflow:**
1. Run Cell 1-4 (setup) every new session — takes ~1 min
2. Run Cell 5-8 (code definitions) — instant
3. Cell 9 (SVM): run once, saves to Drive — skip if already done
4. Cell 10 (CNN training): re-run each session — skips completed folds automatically
5. Cell 11-12 (results): run after all 10 folds complete

**Drive structure it creates:**
```
MyDrive/ECE176_project/
├── UrbanSound8K/          # dataset (download once)
├── checkpoints/           # fold checkpoints + progress.json
├── cache/                 # mfcc.npz (precomputed features)
└── results/               # svm_results.json, cnn_results.json, confusion_matrix.png
```

---

## Architecture Notes

**LightCNN** (v1 CPU): 4 ConvBlocks (1→16→32→64), FC 256, ~300K params  
**UrbanCNN** (v2/v3 GPU): 4 ConvBlocks (1→32→64→128→256), FC 512, ~1.2M params

Both:
- Input: `(B, 1, 128, 128)` log-mel spectrogram (normalized to [0,1])
- Loss: CrossEntropy
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealing
- Augmentation: random time shift ±10%, Gaussian noise σ=0.005

**SVM baseline:**
- Features: 40 MFCCs × (mean + std) = 80-dim vector
- Pipeline: StandardScaler → SVC(rbf, C=10, gamma='scale')

---

## Expected Results

Based on published literature on UrbanSound8K:
- MFCC + SVM: ~70-73% mean accuracy
- CNN from scratch on mel spectrograms: ~75-82% mean accuracy

If CNN underperforms SVM significantly (< 65%), check:
1. Mel length mismatch — verify `MEL_LENGTH` gives valid CNN dimensions
2. Learning rate too high — try `lr=5e-4`
3. Too few epochs — try 80+

---

## Paper Reference

Salamon et al., "A Dataset and Taxonomy for Urban Sound Research", ACM MM 2014  
**Important:** Always use the official 10-fold split. Do NOT shuffle samples across folds.
