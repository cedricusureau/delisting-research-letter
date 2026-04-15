# Delisting research letter — reproduction package

This repository contains the dataset and the scripts
required to reproduce **Figure 1** and **Figure 2** of the research
letter on kidney-transplant delisting strategies.

## Contents

```
public_release/
├── README.md               # this file
├── requirements.txt        # Python dependencies
├── figure1.py              # generates Figure 1 (2 panels)
├── figure2.py              # generates Figure 2 (3 panels)
├── run_gridsearch.py       # optional — rebuilds the grid-search CSVs
├── data/
│   ├── figure1_patients.csv
│   ├── figure1_gridsearch_standard.csv
│   ├── figure1_gridsearch_bootstrap.csv
│   ├── figure2_patients.csv
│   └── figure2_serums.csv
└── outputs/                # created on first run
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the figures

```bash
python figure1.py
python figure2.py
```

Each script writes a `.png` and a `.pdf` into `outputs/`. Running both
takes under a minute on a modern laptop.

## Data description

All times are expressed in **days**.

### `data/figure1_patients.csv` — one row per patient

| column | description |
|---|---|
| `patient_id` | sequential integer |
| `cpra_baseline` | cPRA before delisting (%) |
| `cpra_post` | cPRA after delisting (%) |
| `cpra_fc` | cPRA fold-change: `(100 − cpra_post) / (100 − cpra_baseline)` |
| `time_to_transplant_days` | days from delisting to transplant or end of follow-up |
| `transplant_event` | 1 = transplanted, 0 = censored |

### `data/figure1_gridsearch_standard.csv`

Precomputed standard grid-search output used by Panel A of Figure 1.
Each row is one `(cpra_min, cpra_fc)` threshold combination applied
to the full cohort.

| column | description |
|---|---|
| `cpra_min` | lower cPRA threshold (%) used to select the population |
| `cpra_fc` | cPRA-FC threshold used to split the population into two groups |
| `p_value` | log-rank p-value comparing the two groups |
| `test_statistic` | log-rank test statistic |
| `rate_low` / `rate_high` | 2-year transplant rate (%) in each group |
| `effect_size` | absolute difference between `rate_high` and `rate_low` (percentage points) |
| `n_total`, `n_low`, `n_high` | population sizes (full, low-FC group, high-FC group) |
| `is_significant` | `True` if `p_value ≤ 0.05` |
| `has_effect` | `True` if `effect_size ≥ 5` percentage points |

### `data/figure1_gridsearch_bootstrap.csv`

Bootstrap grid-search output used by Panel A. For each
`(cpra_min, cpra_fc)` combination, the log-rank test is repeated on
100 bootstrap samples (85 % of the cohort, sampled with replacement)
and the stability of the result is summarised across samples.

| column | description |
|---|---|
| `cpra_min`, `cpra_fc` | threshold combination |
| `n_valid_tests` | number of bootstrap iterations (out of 100) that passed minimum population/balance criteria |
| `p_mean`, `p_std`, `p_median`, `p_min`, `p_max` | distribution of p-values across bootstrap samples |
| `significant_rate` | fraction of samples with `p_value ≤ 0.05` — this is the value plotted as the "Bootstrap >70 %" frontier |
| `effect_mean`, `effect_std` | distribution of the 2-year rate difference across samples |
| `population_mean`, `population_std` | distribution of the population size across samples |
| `stability_score` | `1 / (1 + p_std)` — informal measure of how consistent the p-value is across bootstraps |
| `meets_target` | `True` if `significant_rate ≥ 0.66` |

### `data/figure2_patients.csv` — one row per patient

| column | description |
|---|---|
| `patient_id` | sequential integer |
| `dsa_category` | 1 (no DSA >2000 pre-transplant), 2 (historical only), 3 (persistent at D0) |
| `event_composite` | 1 = graft loss or death, 0 = censored |
| `time_composite_days` | days to composite event or end of follow-up |
| `time_followup_days` | overall follow-up time (days), used for censoring corrections |

### `data/figure2_serums.csv` — one row per serum

| column | description |
|---|---|
| `patient_id` | matches `figure2_patients.csv` |
| `days_since_transplant` | days between the serum and the transplant (negative = pre-transplant) |
| `is_day0` | `True` if this serum is the Day-0 serum |
| `mfi_max` | maximum MFI across all donor-specific antibodies in the serum |

Sera are restricted to the window `[-365, +1095]` days around the
transplant — the range effectively used by Panel A.

## Rebuilding the grid-search from scratch (optional)

Figure 1 Panel A relies on two precomputed grid-search CSVs. They are
shipped under `data/` so that `figure1.py` can be run immediately.

If you want to rebuild them end-to-end from `figure1_patients.csv`,
run:

```bash
python run_gridsearch.py
```

This replays the exact same logic used in the paper (logrank tests
over an 80×120 grid, with 100 bootstrap samples per cell for the
bootstrap grid). Expect a long runtime — tens of minutes on a modern
laptop. A `--fast` flag reduces the bootstrap iterations to 25 for a
quicker sanity check, and `--skip-bootstrap` rebuilds only the
standard grid.

## Citation

If you reuse this code or data, please cite the research letter.
*(Full citation to be added once the paper is published.)*
