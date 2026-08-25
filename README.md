# Automated Accuracy Test Bench Software (AATBS)

Python GUI that builds a PlantVillage test set, displays images one-by-one for camera capture, then imports STM32 SD-card inference logs and reports accuracy.

## Setup

```bash
cd Ikmal/AATBS
pyenv local AATBS_env          # already set via .python-version
pip install -r requirements.txt
python main.py
```

## Run folders

Each test run is stored under `data/` as:

```
data/run1_20260812/
  metadata.json          # test-set ground truth
  <sd>.csv               # moved SD inference log (original filename)
  analysis_results.csv   # paired ground-truth vs prediction table (not a copy of the SD file)
  chart.png              # overall + confusion + per-class charts
```

The next folder number is `max(existing runN_*) + 1`.

## Workflow

1. **Setup** — Point at a PlantVillage folder, set images per class, generate a test set → creates `data/runN_YYYYMMDD/metadata.json`.
2. **Capture Display** — Show images one by one. Mark each as captured after the device photographs the screen, or mark as pending to undo a capture.
3. **Analysis** — Point at an SD folder (or a single `.csv`). The folder must contain **exactly one** inference `.csv` (`analysis_results.csv` is ignored). That file is **moved** into the run folder with its original name. A separate `analysis_results.csv` is written as the paired accuracy table. Use **Load old analysis** to reopen a past run.

## SD CSV format

Expected columns (from firmware `capture_sd_log.cpp`):

```
id,gate_pred,pct_leaf,pct_others,pct_pest,disease_pred,gate_ms,infer_ms,bmp
```

A tiny example file is in `sample_sd/log.csv`.

## Notes

- With “Normalize folders” enabled, PlantVillage folder names are mapped toward `bacterial / fungal / healthy / pest / viral` to match the ACLIS disease model.
- Capture order must match SD `id` order (image 1 → id 1, and so on).
