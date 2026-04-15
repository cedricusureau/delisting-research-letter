"""
run_gridsearch.py — Reproduce the two grid-search CSVs used by Figure 1
from figure1_patients.csv.

OPTIONAL. Figure 1 can already be generated directly from the precomputed
grids shipped in data/. This script exists for full reproducibility: it
rebuilds those grids from the patient-level data using the same
thresholds and bootstrap settings as the original study.

Outputs:
    data/figure1_gridsearch_standard.csv
    data/figure1_gridsearch_bootstrap.csv

Runtime: long. The bootstrap grid scans ~15000 threshold combinations,
each evaluated on 100 bootstrap samples. Expect tens of minutes on a
modern laptop. Use --fast to reduce the bootstrap iterations (25 instead
of 100) for a quicker sanity check.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

CONFIG = {
    "patients_csv": DATA_DIR / "figure1_patients.csv",
    "output_standard": DATA_DIR / "figure1_gridsearch_standard.csv",
    "output_bootstrap": DATA_DIR / "figure1_gridsearch_bootstrap.csv",
    "cpra_range": np.arange(94.0, 100.01, 0.04),
    "cpra_fc_range_standard": np.arange(1.0, 7.01, 0.05),
    "cpra_fc_range_bootstrap": np.arange(1.0, 10.01, 0.05),
    "cpra_max": 100,
    "min_patients_total": 50,
    "min_patients_per_group": 15,
    "min_balance_pct": 15,
    "significance_threshold": 0.05,
    "min_effect_size": 5,
    "n_bootstrap": 100,
    "bootstrap_fraction": 0.85,
    "target_significance_rate": 0.66,
    "random_seed": 42,
}


def load_patients():
    df = pd.read_csv(CONFIG["patients_csv"])
    df["cpra_fc_abs"] = df["cpra_fc"].abs()
    return df


def population_subset(df, cpra_min, cpra_max):
    return df[(df["cpra_baseline"] >= cpra_min) & (df["cpra_baseline"] <= cpra_max)]


def rate_at_2_years(time_days, event, threshold_days=730):
    if len(time_days) == 0:
        return np.nan
    transplanted = ((time_days <= threshold_days) & (event == 1)).sum()
    with_followup = ((time_days >= threshold_days) | (event == 1)).sum()
    if with_followup == 0:
        return np.nan
    return transplanted / with_followup * 100


def single_logrank(population, cpra_fc_threshold):
    if len(population) < CONFIG["min_patients_total"]:
        return None

    low = population[population["cpra_fc_abs"] < cpra_fc_threshold]
    high = population[population["cpra_fc_abs"] >= cpra_fc_threshold]

    n_low, n_high = len(low), len(high)
    total = n_low + n_high

    if n_low < CONFIG["min_patients_per_group"] or n_high < CONFIG["min_patients_per_group"]:
        return None
    if min(n_low, n_high) / total * 100 < CONFIG["min_balance_pct"]:
        return None

    try:
        lr = logrank_test(
            low["time_to_transplant_days"], high["time_to_transplant_days"],
            low["transplant_event"], high["transplant_event"],
        )
    except Exception:
        return None

    rate_low = rate_at_2_years(low["time_to_transplant_days"], low["transplant_event"])
    rate_high = rate_at_2_years(high["time_to_transplant_days"], high["transplant_event"])
    if np.isnan(rate_low) or np.isnan(rate_high):
        effect = 0.0
    else:
        effect = abs(rate_high - rate_low)

    return {
        "p_value": lr.p_value,
        "test_statistic": lr.test_statistic,
        "rate_low": rate_low,
        "rate_high": rate_high,
        "effect_size": effect,
        "n_low": n_low,
        "n_high": n_high,
        "n_total": total,
        "is_significant": lr.p_value <= CONFIG["significance_threshold"],
        "has_effect": effect >= CONFIG["min_effect_size"],
    }


def run_standard_grid(df):
    print(f"Standard grid: {len(CONFIG['cpra_range'])} × "
          f"{len(CONFIG['cpra_fc_range_standard'])} combinations")
    rows = []
    for cpra_min in CONFIG["cpra_range"]:
        cpra_min = round(float(cpra_min), 2)
        pop = population_subset(df, cpra_min, CONFIG["cpra_max"])
        for cpra_fc in CONFIG["cpra_fc_range_standard"]:
            cpra_fc = round(float(cpra_fc), 2)
            result = single_logrank(pop, cpra_fc)
            if result is not None:
                rows.append({"cpra_min": cpra_min, "cpra_fc": cpra_fc, **result})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(CONFIG["output_standard"], index=False)
    print(f"  → {CONFIG['output_standard'].name}: {len(df_out)} rows")
    return df_out


def run_bootstrap_grid(df, n_bootstrap):
    print(f"Bootstrap grid: {len(CONFIG['cpra_range'])} × "
          f"{len(CONFIG['cpra_fc_range_bootstrap'])} combinations × "
          f"{n_bootstrap} bootstrap samples")
    rng_base = CONFIG["random_seed"]
    rows = []
    total_pairs = len(CONFIG["cpra_range"]) * len(CONFIG["cpra_fc_range_bootstrap"])
    pair_idx = 0

    for cpra_min in CONFIG["cpra_range"]:
        cpra_min = round(float(cpra_min), 2)
        for cpra_fc in CONFIG["cpra_fc_range_bootstrap"]:
            cpra_fc = round(float(cpra_fc), 2)
            pair_idx += 1

            if pair_idx % max(1, total_pairs // 20) == 0:
                print(f"  progress {pair_idx}/{total_pairs} "
                      f"({pair_idx / total_pairs * 100:.0f}%)")

            p_values, effects, sizes = [], [], []
            for b in range(n_bootstrap):
                sample = df.sample(
                    frac=CONFIG["bootstrap_fraction"],
                    replace=True,
                    random_state=rng_base + b + pair_idx * 1000,
                )
                pop = population_subset(sample, cpra_min, CONFIG["cpra_max"])
                result = single_logrank(pop, cpra_fc)
                if result is not None:
                    p_values.append(result["p_value"])
                    effects.append(result["effect_size"])
                    sizes.append(result["n_total"])

            valid = len(p_values)
            if valid < n_bootstrap * 0.7:
                continue

            p_arr = np.asarray(p_values)
            e_arr = np.asarray(effects)
            s_arr = np.asarray(sizes)
            sig_rate = float(np.mean(p_arr <= CONFIG["significance_threshold"]))

            rows.append({
                "cpra_min": cpra_min,
                "cpra_fc": cpra_fc,
                "n_valid_tests": valid,
                "p_mean": float(p_arr.mean()),
                "p_std": float(p_arr.std()),
                "p_median": float(np.median(p_arr)),
                "p_min": float(p_arr.min()),
                "p_max": float(p_arr.max()),
                "significant_rate": sig_rate,
                "effect_mean": float(e_arr.mean()),
                "effect_std": float(e_arr.std()),
                "population_mean": float(s_arr.mean()),
                "population_std": float(s_arr.std()),
                "stability_score": float(1 / (1 + p_arr.std())),
                "meets_target": sig_rate >= CONFIG["target_significance_rate"],
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(CONFIG["output_bootstrap"], index=False)
    print(f"  → {CONFIG['output_bootstrap'].name}: {len(df_out)} rows")
    return df_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true",
        help="Use 25 bootstrap iterations instead of 100 (faster, less stable).",
    )
    parser.add_argument(
        "--skip-bootstrap", action="store_true",
        help="Skip the bootstrap grid (only rebuild the standard grid).",
    )
    args = parser.parse_args()

    df = load_patients()
    print(f"Loaded {len(df)} patients from {CONFIG['patients_csv'].name}")

    run_standard_grid(df)
    if not args.skip_bootstrap:
        n_bootstrap = 25 if args.fast else CONFIG["n_bootstrap"]
        run_bootstrap_grid(df, n_bootstrap)

    print("Done. You can now rerun figure1.py.")


if __name__ == "__main__":
    main()
