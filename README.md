# VinDatathon Forecasting Pipeline

End-to-end analytics and forecasting project for the VinDatathon retail sales challenge. The repository combines exploratory business analysis, reproducible model training, rolling-origin validation, explainability plots, and final submission artifacts for daily Revenue and COGS prediction.

The final model is built around calendar/event feature engineering, annual and weekly seasonality, Vietnamese holiday effects, promotion windows, quarterly specialists, and margin calibration for recurring COGS behavior.

---

## Project Status

- Final submission file included at `submission/submission.csv`.
- Main production script: `src/final_forecasting_pipeline.py`.
- Validation experiment script: `src/oof_rolling_experiment.py`.
- Report and exploratory figures are stored under `EDA/` and `figures/`.
- The final pipeline was re-run successfully and passed the competition submission format checks.

---

## Repository Structure

```text
vin_datathon-main/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- raw/                         # Expected raw competition data files
|-- EDA/
|   |-- main.tex                     # LaTeX report source
|   |-- saved_*.png                  # Exploratory analysis charts
|   `-- qa_pages/                    # Supporting report/QA assets
|-- figures/
|   |-- generate_figures.py          # Rebuilds model and validation figures
|   |-- calibration_search.png
|   |-- cogs_ratio.png
|   |-- feature_importance.png
|   |-- final_forecast.png
|   |-- model_pipeline.png
|   `-- validation_curve.png
|-- src/
|   |-- final_forecasting_pipeline.py # Full final training and submission pipeline
|   `-- oof_rolling_experiment.py     # Rolling-origin validation experiment
`-- submission/
    `-- submission.csv               # Final submitted forecast artifact
```

Generated runtime outputs are written by the scripts to `outputs/submissions/`. This folder is created automatically when the pipelines run.

---

## Data Requirements

The forecasting scripts expect competition files in `data/raw/`, especially:

- `sales.csv` with at least `Date`, `Revenue`, and `COGS` columns.
- `sample_submission.csv` with the required forecast dates and submission schema.

If `sample_submission.csv` is not available, `src/final_forecasting_pipeline.py` falls back to generating dates from `2023-01-01` through `2024-07-01`, but the official format check requires the real sample file.

---

## Environment Setup

Use Python 3.11+ if possible, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional dependency:

```bash
pip install prophet
```

`prophet` is optional. The final pipeline detects whether it is installed and uses a fallback path when it is unavailable.

---

## How To Run

Recommended order:

```bash
pip install -r requirements.txt
python src/oof_rolling_experiment.py
python src/final_forecasting_pipeline.py
python figures/generate_figures.py
```

### 1. Run the Final Forecasting Pipeline

```bash
python src/final_forecasting_pipeline.py
```

This script:

- Loads raw sales data from `data/raw/`.
- Builds leakage-safe calendar, Fourier, Tet, holiday, promotion, and odd-year regime features.
- Trains Ridge, LightGBM, XGBoost, Holt-Winters, optional Prophet, and quarter-specialist models.
- Blends predictions and applies COGS margin calibration.
- Writes final and diagnostic artifacts to `outputs/submissions/`.
- Verifies row count, column order, date order, and null values against `data/raw/sample_submission.csv`.

Key output:

```text
outputs/submissions/submission.csv
```

Latest verified run summary:

| Check | Result |
|---|---|
| Training range | 2012-07-04 to 2022-12-31 |
| Forecast horizon | 548 rows |
| Feature count | 82 |
| Final average Revenue | 4,181,713 |
| Final average COGS | 3,846,614 |
| Format validation | Passed |

### 2. Run Rolling-Origin Validation

```bash
python src/oof_rolling_experiment.py
```

This script stress-tests the forecasting approach across validation years and writes:

```text
outputs/submissions/oof_rolling_experiment_results.csv
```

### 3. Regenerate Figures

After running the validation experiment and final pipeline, rebuild the summary figures:

```bash
python figures/generate_figures.py
```

This updates plots in `figures/`, including the pipeline diagram, calibration search, COGS ratio chart, validation curve, and final forecast trajectory.

---

## Modeling Approach

The project uses a calibrated ensemble designed for daily retail forecasting:

| Component | Purpose |
|---|---|
| Calendar and event features | Capture day-of-week, month-end, quarter, Tet, Vietnamese holidays, and shopping events. |
| Fourier terms | Encode annual, weekly, and monthly seasonality smoothly. |
| Ridge regression | Provides a stable linear baseline on normalized feature space. |
| LightGBM | Main nonlinear learner for Revenue and COGS. |
| Quarter specialists | Reweights quarter-specific observations to improve seasonal fit. |
| XGBoost | Adds a second gradient-boosting learner for ensemble diversity. |
| Holt-Winters | Supplies a statistical seasonal anchor from recent history. |
| Optional Prophet | Adds another trend/seasonality component when installed. |
| Margin calibration | Adjusts COGS using odd/even year quarterly COGS-to-Revenue ratios. |

---

## Feature Engineering Highlights

- Calendar features: year, month, day, day-of-week, day-of-year, quarter, weekends, and month boundaries.
- Seasonality: annual, weekly, and monthly sine/cosine Fourier features.
- Vietnamese holidays: New Year, Reunification Day, Labor Day, National Day, Tet windows, 11.11, 12.12, Christmas, and Black Friday.
- Promotion windows: spring sale, mid-year sale, fall launch, year-end sale, urban blowout, and rural special.
- Regime indicators: pre-2019, 2019 transition, post-2019, and odd-year behavior.
- Leakage control: validation code trains fold models only on data available before each validation period.

---

## Outputs

Depending on which scripts are run, expected generated outputs include:

| Path | Description |
|---|---|
| `outputs/submissions/submission.csv` | Main generated final submission. |
| `outputs/submissions/submission_upgraded.csv` | Diagnostic upgraded candidate. |
| `outputs/submissions/validation_report.csv` | Holdout validation metrics. |
| `outputs/submissions/oof_rolling_experiment_results.csv` | Rolling-origin experiment results. |
| `outputs/submissions/shap_revenue_summary.png` | SHAP summary for Revenue model. |
| `outputs/submissions/shap_cogs_summary.png` | SHAP summary for COGS model. |
| `outputs/submissions/feature_importance.png` | LightGBM feature importance chart. |
| `figures/*.png` | Presentation-ready model and validation figures. |

The checked-in final forecast is available at `submission/submission.csv`.

Runtime outputs are intentionally separated from the checked-in `submission/` folder so generated experiments do not overwrite the submitted artifact unless copied deliberately.

---

## Reproducibility Notes

- Random seed is fixed to `42` in the modeling scripts.
- `outputs/submissions/` is created automatically by the Python scripts.
- The final pipeline performs a submission format check before completion.
- Some diagnostic code expects previously generated files, so run the final pipeline before figure generation.

---

## Quick Commands

```bash
pip install -r requirements.txt
python src/oof_rolling_experiment.py
python src/final_forecasting_pipeline.py
python figures/generate_figures.py
```

To inspect the final generated submission quickly:

```bash
python -c "import pandas as pd; print(pd.read_csv('outputs/submissions/submission.csv').head())"
```

---

## Project Goal

Deliver an accurate, explainable daily forecast for Revenue and COGS while connecting model behavior to business patterns found during exploratory analysis, including seasonal demand, promotion effects, holiday behavior, and recurring margin/COGS regimes.