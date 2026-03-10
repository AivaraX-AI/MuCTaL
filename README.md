# MuCTaL

**A lightweight multi-cancer tumor localization framework for whole-slide histopathology images (H&E).**

MuCTaL provides scripts and notebooks for end-to-end pathology AI workflows:

- whole-slide preprocessing and tiling,
- tile-level model training and inference (FastAI/PyTorch),
- post-processing into clinically useful outputs (heatmaps and GeoJSON regions),
- cross-validation utilities and exploratory notebooks.

Trained model available on huggingface https://huggingface.co/hillmancancercenterds/MuCTaL

![Alt text](/tile_prediction_heatmap.png)



---

## Repository Summary

This repository is organized around a practical WSI (whole slide image) pipeline:

1. **Preprocess slides into tissue-rich tiles** using `pathml`-based pipelines.
2. **Train CNN models** on labeled tiles (FastAI).
3. **Run inference** on unseen tiles/slides.
4. **Convert predictions** to visual artifacts (heatmaps) and annotation formats (GeoJSON).
5. **Evaluate folds/repeats** with cross-validation helper scripts.

The codebase appears to be research/HPC-oriented and includes path placeholders and batch-style scripts, so users should expect to adapt data paths and execution wrappers to their environment.

---

## Folder Structure

```text
MuCTaL/
├── LICENSE
├── README.md
├── helpers/
│   ├── __init__.py
│   ├── anno.py
│   ├── preproc.py
│   └── tile.py
├── notebooks/
│   ├── 01_generate_wsi_samplesheet_run_preprocessing.ipynb
│   ├── 02_annotated_tile_file_org_for_training.ipynb
│   ├── 03_train_model_fastai2.7.ipynb
│   ├── 04_model_eval_fastai2.7.ipynb
│   ├── 05_example_inference_to_geojson.ipynb
│   ├── 06_acral_tile_heatmap_class_viz.ipynb
│   └── 07_percent_predicted_tumor_each_slide.ipynb
├── pipeline/
│   ├── fastai_inference_v10.py
│   ├── pathml_preproc_v10.py
│   ├── tile_infer_to_geojson.py
│   └── tile_infer_to_heatmap.py
└── train/
    └── train_full.py
```

---

## What Each Main Module Does

- **`pipeline/`**: Main runnable pipeline scripts for preprocessing, inference, and output generation.
- **`train/`**: End-to-end model training script for full dataset training.
- **`helpers/`**: Reusable utility code (annotation geometry checks, preprocessing helpers, tile parsing).
- **`notebooks/`**: Interactive analysis/tutorial notebooks for data prep, training, evaluation, and visualization.

---

## Typical Workflow

1. **Prepare sample metadata** (notebooks and TSV inputs).
2. **Preprocess WSIs to tiles** with `pipeline/pathml_preproc_v10.py`.
3. **Build balanced tile CSVs** 
4. **Train model(s)** with scripts in `train/` 
5. **Infer tile probabilities** using `pipeline/fastai_inference_v10.py`.
6. **Generate outputs**:
   - GeoJSON tumor regions: `pipeline/tile_infer_to_geojson.py`
   - Slide heatmaps/overlays: `pipeline/tile_infer_to_heatmap.py`

---

## Requirements

Core dependencies inferred from scripts:

- Python 3.9+
- `fastai`, `torch`, `torchvision`
- `pathml`
- `opencv-python`
- `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`
- `dask`, `distributed` (optional, for distributed preprocessing)
- `Pillow`
- `cv2geojson`

> Many scripts are designed for HPC/SLURM environments and may rely on environment variables (e.g., `SLURM_SCRATCH`) and local data layouts.

---

## Quick Start

> Update all paths below for your environment.

### 1) Preprocess a slide to tiles

```bash
python pipeline/pathml_preproc_v10.py \
  /path/to/output \
  /path/to/slide.svs \
  224 \
  /path/to/MuCTaL
```

### 2) Run tile inference

```bash
python pipeline/fastai_inference_v10.py \
  /path/to/tiles_df.tsv \
  /path/to/model.pkl \
  /path/to/tile_root/ \
  /path/to/output
```

### 3) Convert predictions to GeoJSON

```bash
python pipeline/tile_infer_to_geojson.py \
  /path/to/infer_tiles.tsv \
  /path/to/output_geojson
```

### 4) Generate heatmap overlay

```bash
python pipeline/tile_infer_to_heatmap.py \
  /path/to/infer_tiles.tsv \
  /path/to/original_slide.svs \
  /path/to/output_heatmaps
```

---

## Notes and Caveats

- Several scripts contain hard-coded placeholders (e.g., `/path/to/...`) and versioned naming conventions.
- Some utilities expect specific input TSV schemas (tile paths, slide/case IDs, class labels).
- Cross-validation scripts assume specific model naming patterns such as:
  `arch_kfold_nrep_nbal_px_v` (example style from code).

---

## License

This project includes a `LICENSE` file in the repository root. See that file for usage terms.

