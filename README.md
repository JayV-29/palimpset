# PALIMPSEST — Identified Counterfactual Explanations for Load Forecasting

One Kaggle notebook (`palimpsest.ipynb`) that runs the whole pipeline and every experiment:

| Section | Content |
|---|---|
| Data | EIA-930 hourly CAISO demand, NOAA ISD hourly weather (LAX, Sacramento, San Jose, Fresno), CAISO Flex Alert dates, Low Carbon London (Kaggle mirror `jeanmidev/smart-meters-in-london`) |
| Model | Thermal layer (degree-hours with thermal-mass lag, NLS) → behavioral layer identified from matched natural-experiment contrasts → LightGBM quantile residual forecaster with conformalized intervals |
| Counterfactuals | Abduction–action–prediction, kNN plausibility projection, `thermal + behavioral + residual` decomposition, analog-day spread |
| Eval A | Accuracy on 2023–2024 vs plain LightGBM / persistence / similar-day |
| Eval B (central) | Interventional holdout on Flex Alert days after 2021: identified vs full-series ablation vs flag-as-feature |
| Eval C | COVID lockdown holdout ("same weather, 2020 occupancy") |
| Eval D | Low Carbon London ToU response transferred across household halves |
| Interactive | ipywidgets view: choose a day, pose ΔT / appeal, see decomposition + analogs |

## Running on Kaggle

1. Create a new notebook, upload `palimpsest.ipynb` (or `kaggle kernels push` with `kernel-metadata.json`).
2. Settings → **Internet: On** (for EIA-930 and NOAA downloads, ~1 GB cached under `/kaggle/working/cache`).
3. Add data → search **Smart meters in London** (`jeanmidev/smart-meters-in-london`) for Eval D.
   Optionally add the UK Data Service `Tariffs.csv` (TariffDateTime, Tariff) to estimate effects per price state.
4. Run all. CPU only; a few minutes of compute plus download time.

If any source is unavailable, the affected experiment automatically falls back to **synthetic data with a
known ground truth** (the synthetic thermal term deliberately contains humidity and saturation physics the
model does not have). The summary cell states which mode was used.

Set `PALIMPSEST_FORCE_SYNTHETIC=1` to force synthetic mode; `LCL_MAX_BLOCKS` caps how many hhblock files are read.

## Local development

```
uv venv .venv && uv pip install -p .venv/bin/python numpy pandas scipy scikit-learn lightgbm nbformat matplotlib requests jupyter ipywidgets
.venv/bin/python build_notebook.py                # notebook_src.py (# %% cells) -> palimpsest.ipynb
PALIMPSEST_FORCE_SYNTHETIC=1 .venv/bin/jupyter nbconvert --to notebook --execute palimpsest.ipynb --output executed.ipynb
```

`notebook_src.py` is the source of truth; edit it and rebuild.

## Flex Alert dates

`FLEX_ALERT_DATES` in the notebook was compiled from CAISO / flexalert.org public notices (2016–2022). Treat it
as data: verify or replace it with the authoritative archive before drawing conclusions from real-data runs.

## Image credits

Header photographs are from Wikimedia Commons under CC licenses; attributions are printed under each image.
