# Retinal DR Hybrid Pipeline (ViT + EfficientNet + PSO + Fuzzy)

This project is a starter implementation created from the document:
`AdityaMittal_2210990069 (2).docx`.

It follows the proposed pipeline in your report:
1. Image preprocessing and augmentation.
2. Dual feature extraction with EfficientNet (local features) and ViT (global features).
3. Feature fusion.
4. PSO-based feature selection (with a fallback selector if `pyswarms` is not installed).
5. Fuzzy logic classifier for DR severity grading.

## Files

- `dr_hybrid_pipeline.py`: Main source code.
- `README.md`: Setup and usage guide.

## Dataset Layout

Organize your data as class folders (ImageFolder format):

```text
data/
  train/
    0/
    1/
    2/
    3/
    4/
  val/
    0/
    1/
    2/
    3/
    4/
```

Where class labels map to DR grades (example):
- `0`: No DR
- `1`: Mild
- `2`: Moderate
- `3`: Severe
- `4`: Proliferative DR

## Setup

Use Python 3.10+ and install dependencies:

```bash
pip install torch torchvision scikit-learn numpy
```

Optional (for actual PSO instead of fallback):

```bash
pip install pyswarms
```

## Run

```bash
python dr_hybrid_pipeline.py --data-dir data --epochs 3 --batch-size 8 --num-classes 5
```

Useful options:
- `--image-size 224`
- `--fused-dim 512`
- `--selected-features 256`
- `--out-dir artifacts`

## Output Artifacts

After running, outputs are saved under `artifacts/`:
- `hybrid_feature_model.pt`
- `selected_feature_indices.npy`
- `run_summary.json`

## Notes

- This is a practical research baseline, not a clinical tool.
- For reproducible experiments, keep train/val split fixed and report metrics such as accuracy, F1, sensitivity, and specificity.
- You can extend the fuzzy block with richer linguistic rules if you want closer alignment with your final thesis version.
