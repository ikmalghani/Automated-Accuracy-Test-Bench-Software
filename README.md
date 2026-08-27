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
  metadata.json            # test-set ground truth
  <sd>.csv                 # copied SD inference log (original filename)
  analysis_results.csv     # paired ground-truth vs prediction table
  chart_gate.png           # gate model: overall + confusion + per-class
  chart_classifier.png     # disease CNN only (SKIP excluded)
  chart_pipeline.png       # overall cascade (current pipeline analysis)
  chart.png                # copy of chart_pipeline.png (legacy name)
```

The next folder number is `max(existing runN_*) + 1`.

## Workflow

1. **Setup** — Point at a PlantVillage folder, set images per class, generate a test set → creates `data/runN_YYYYMMDD/metadata.json`.
2. **Capture Display** — Show images one by one. Mark each as captured after the device photographs the screen, or mark as pending to undo a capture.
3. **Analysis** — Point at an SD folder (or a single `.csv`). The folder must contain **exactly one** inference `.csv` (`analysis_results.csv` is ignored). That file is **copied** into the run folder with its original name. A separate `analysis_results.csv` is written as the paired accuracy table. Three chart images are written. Use **Load old analysis** to reopen a past run.

   The Analysis tab has three views:

   - **Gate model** — `gate_pred` vs folder ground truth in gate label space. Reports how many classes the gate has (2-class Better Gate: `leaf` / `not_leaf`; 3-class Leaf+Pest: `leaf` / `others` / `pest`), overall accuracy, per-class accuracy, and a confusion matrix.
   - **Plant disease classifier** — CNN `disease_pred` vs folder ground truth. Rows where the disease result is `SKIP` are ignored.
   - **Overall pipeline** — the previous whole-cascade score, including gate `SKIP`s.


## SD CSV format

Expected columns (from firmware `capture_sd_log.cpp`):

```
id,gate_pred,pct_leaf,pct_not_leaf,disease_pred,gate_ms,infer_ms,bmp
```

Older Leaf+Pest logs with `pct_others` / `pct_pest` still import.

A tiny example file is in `sample_sd/log.csv`.

## Notes

- With “Normalize folders” enabled, PlantVillage folder names are mapped toward `bacterial / fungal / healthy / pest / viral` to match the ACLIS disease model.
- Capture order must match SD `id` order (image 1 → id 1, and so on).
- **Gate accuracy** maps folder names into the gate's classes (detected from the log):
  - 2-class: disease folders including `pest` → `leaf`; `others` / `not_leaf` folders → `not_leaf`
  - 3-class: `bacterial` / `fungal` / `healthy` / `viral` → `leaf`; `pest` → `pest`; `others` / `not_leaf` → `others`
- **Classifier accuracy** ignores `SKIP` and compares the CNN label to the folder name.
- **Pipeline accuracy** counts every paired image, including gate `SKIP`s:
  - folder `others` / `non_leaf` / `not_leaf`: correct if the gate rejects (`not_leaf` / `others`)
  - disease folders including `pest`: a gate reject/SKIP is a miss; otherwise the disease label must match the folder
