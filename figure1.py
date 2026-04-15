"""
figure1.py — Figure 1 of the research letter (2 panels)

Panel A: Significance zones — standard vs bootstrap thresholds
         across a (cPRA, cPRA-FC) grid.
Panel B: Kaplan–Meier at the selected operating point (cPRA ≥ 98 %,
         cPRA-FC ≥ 3.0), split on cPRA-FC.

Inputs (data/):
    figure1_patients.csv            (patient-level data)
    figure1_gridsearch_standard.csv (precomputed standard grid)
    figure1_gridsearch_bootstrap.csv (precomputed bootstrap grid)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"

CONFIG = {
    "bootstrap_csv": DATA_DIR / "figure1_gridsearch_bootstrap.csv",
    "standard_csv": DATA_DIR / "figure1_gridsearch_standard.csv",
    "patients_csv": DATA_DIR / "figure1_patients.csv",
    "figure_size": (16, 7),
    "dpi": 300,
    "operating_point": {"cpra": 98.0, "cpra_fc": 3.0, "label": "B"},
    "km_colors": ["#d62728", "#2ca02c"],
    "title_fontsize": 14,
    "label_fontsize": 12,
    "annotation_fontsize": 10,
    "smoothing_sigma": 1.5,
    "bootstrap_threshold": 0.70,
    "significance_threshold": 0.05,
}

plt.rcParams.update({"font.size": 11})


def load_patients():
    df = pd.read_csv(CONFIG["patients_csv"])
    df["cpra_fc_abs"] = df["cpra_fc"].abs()
    return df


def load_standard_grid():
    df = pd.read_csv(CONFIG["standard_csv"])
    df = df[(df["is_significant"] == True) & (df["has_effect"] == True)].copy()
    if df.empty:
        return None

    cpra_values = sorted(df["cpra_min"].unique())
    fc_values = sorted(df["cpra_fc"].unique())
    pvalues = np.full((len(cpra_values), len(fc_values)), 1.0)

    for i, cpra in enumerate(cpra_values):
        for j, fc in enumerate(fc_values):
            row = df[(df["cpra_min"] == cpra) & (df["cpra_fc"] == fc)]
            if len(row) > 0:
                pvalues[i, j] = row.iloc[0]["p_value"]

    log_p = -np.log10(np.clip(pvalues, 1e-10, 1.0))
    smoothed = 10 ** (-gaussian_filter(log_p, sigma=CONFIG["smoothing_sigma"]))

    records = [
        {"cpra_min": cpra, "cpra_fc": fc, "p_value_smoothed": smoothed[i, j]}
        for i, cpra in enumerate(cpra_values)
        for j, fc in enumerate(fc_values)
    ]
    return pd.DataFrame(records).pivot_table(
        index="cpra_min", columns="cpra_fc",
        values="p_value_smoothed", aggfunc="first", fill_value=1.0,
    )


def load_bootstrap_grid():
    df = pd.read_csv(CONFIG["bootstrap_csv"])
    return df.pivot_table(
        index="cpra_min", columns="cpra_fc",
        values="significant_rate", aggfunc="first",
    )


def _significant_cpra_range(pivot, threshold, comparator):
    fc_values = sorted(pivot.columns)
    min_vals, max_vals = [], []
    for fc in fc_values:
        col = pivot[fc]
        mask = comparator(col, threshold)
        if mask.any():
            significant = col[mask].index
            min_vals.append(min(significant))
            max_vals.append(max(significant))
        else:
            min_vals.append(None)
            max_vals.append(None)
    return fc_values, min_vals, max_vals


def create_significance_zones_panel(ax, pivot_standard, pivot_bootstrap):
    if pivot_standard is None or pivot_bootstrap is None:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    std_fc, std_min, std_max = _significant_cpra_range(
        pivot_standard, CONFIG["significance_threshold"], lambda c, t: c < t,
    )
    boot_fc, boot_min, boot_max = _significant_cpra_range(
        pivot_bootstrap, CONFIG["bootstrap_threshold"], lambda c, t: c > t,
    )

    valid_std = [i for i, v in enumerate(std_min) if v is not None]
    valid_std_max = [i for i, v in enumerate(std_max) if v is not None]
    valid_boot = [i for i, v in enumerate(boot_min) if v is not None]

    if not (valid_std and valid_std_max and valid_boot):
        return

    fc_std = [std_fc[i] for i in valid_std]
    cpra_std_min = [std_min[i] for i in valid_std]
    fc_std_max = [std_fc[i] for i in valid_std_max]
    cpra_std_max = [std_max[i] for i in valid_std_max]
    fc_boot = [boot_fc[i] for i in valid_boot]
    cpra_boot_min = [boot_min[i] for i in valid_boot]
    cpra_boot_max = [boot_max[i] for i in valid_boot]

    x_common = np.linspace(
        max(min(fc_std), min(fc_boot)),
        min(max(fc_std), max(fc_boot)),
        100,
    )

    f_std = interp1d(fc_std, cpra_std_min, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
    f_std_max = interp1d(fc_std_max, cpra_std_max, kind="linear",
                         bounds_error=False, fill_value="extrapolate")
    f_boot_min = interp1d(fc_boot, cpra_boot_min, kind="linear",
                          bounds_error=False, fill_value="extrapolate")
    f_boot_max = interp1d(fc_boot, cpra_boot_max, kind="linear",
                          bounds_error=False, fill_value="extrapolate")

    y_std = f_std(x_common)
    y_std_max = f_std_max(x_common)
    y_boot_min = f_boot_min(x_common)
    y_boot_max = f_boot_max(x_common)

    valid_mask = (
        (y_std >= 96) & (y_std <= 100)
        & (y_std_max >= 96) & (y_std_max <= 100)
        & (y_boot_min >= 96) & (y_boot_min <= 100)
        & (y_boot_max >= 96) & (y_boot_max <= 100)
    )

    x_plot = x_common[valid_mask]
    y_std_p = y_std[valid_mask]
    y_std_max_p = y_std_max[valid_mask]
    y_boot_min_p = y_boot_min[valid_mask]
    y_boot_max_p = y_boot_max[valid_mask]

    if len(x_plot) == 0:
        return

    ax.fill_between(x_plot, 99.9, y_std_p, alpha=0.5, color="red",
                    label="Non-significant")
    ax.fill_between(x_plot, y_std_max_p, 95.5, alpha=0.5, color="red")
    ax.fill_between(x_plot, y_boot_max_p, y_std_p, alpha=0.8, color="gray",
                    label="Grey zone")
    ax.fill_between(x_plot, y_std_max_p, y_boot_min_p, alpha=0.8, color="gray")
    ax.fill_between(x_plot, y_boot_min_p, y_boot_max_p, alpha=0.7, color="green",
                    label="High confidence")

    ax.plot(fc_std, cpra_std_min, "k-", linewidth=3,
            label="Standard p<0.05", zorder=10)
    ax.plot(fc_std_max, cpra_std_max, "k-", linewidth=3, alpha=0.7, zorder=10)
    ax.plot(fc_boot, cpra_boot_min, "orange", linewidth=3, linestyle="--",
            label="Bootstrap >70%", zorder=10)
    ax.plot(fc_boot, cpra_boot_max, "orange", linewidth=3, linestyle="--", zorder=10)

    ax.scatter(fc_std, cpra_std_min, color="black", s=20, zorder=11, alpha=0.5)
    ax.scatter(fc_std_max, cpra_std_max, color="black", s=20, alpha=0.4, zorder=11)
    ax.scatter(fc_boot, cpra_boot_min, color="orange", s=20, zorder=11, alpha=0.5)
    ax.scatter(fc_boot, cpra_boot_max, color="darkorange", s=20, zorder=11, alpha=0.5)

    min_idx = int(np.argmin(cpra_boot_min))
    min_cpra = cpra_boot_min[min_idx]
    min_fc = fc_boot[min_idx]
    ax.scatter([min_fc], [min_cpra], marker="*", s=250,
               facecolor="black", edgecolor="black", linewidth=1, zorder=25,
               label=f"Minimal cPRA ({min_cpra:.1f}%)")

    ax.set_xlabel("cPRA-FC Threshold", fontsize=CONFIG["label_fontsize"], fontweight="bold")
    ax.set_ylabel("cPRA Baseline (%)", fontsize=CONFIG["label_fontsize"], fontweight="bold")
    ax.set_title("A. Significance Zones: Standard vs Bootstrap Thresholds",
                 fontsize=CONFIG["title_fontsize"], fontweight="bold", pad=10)
    ax.set_xlim(1.25, 6.0)
    ax.set_ylim(99.9, 95.5)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=1)

    op = CONFIG["operating_point"]
    ax.scatter([op["cpra_fc"]], [op["cpra"]], marker="o", s=80,
               facecolor="black", edgecolor="black", zorder=20)
    ax.text(op["cpra_fc"] + 0.08, op["cpra"] - 0.08, op["label"],
            ha="left", va="top", fontsize=11, fontweight="bold",
            color="black", zorder=21)
    ax.legend(loc="upper left", fontsize=9, ncol=1)


def create_kaplan_meier_panel(ax, patients, cpra_threshold, cpra_fc_threshold,
                              panel_label="B"):
    pop = patients[patients["cpra_baseline"] >= cpra_threshold].copy()
    if pop.empty:
        ax.text(0.5, 0.5, f"No patients with cPRA ≥{cpra_threshold}%",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        return

    low = pop[pop["cpra_fc_abs"] < cpra_fc_threshold]
    high = pop[pop["cpra_fc_abs"] >= cpra_fc_threshold]

    if low.empty or high.empty:
        ax.text(0.5, 0.5, "Insufficient data\n(Need both cPRA-FC groups)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        return

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    kmf_low.fit(low["time_to_transplant_days"],
                event_observed=low["transplant_event"],
                label=f"cPRA-FC <{cpra_fc_threshold} (n={len(low)})")
    kmf_high.fit(high["time_to_transplant_days"],
                 event_observed=high["transplant_event"],
                 label=f"cPRA-FC ≥{cpra_fc_threshold} (n={len(high)})")

    prob_low = 1 - kmf_low.survival_function_.iloc[:, 0]
    prob_high = 1 - kmf_high.survival_function_.iloc[:, 0]

    ax.step(kmf_low.timeline, prob_low, where="post",
            color=CONFIG["km_colors"][0], linewidth=2.5, alpha=0.8,
            label=f"cPRA-FC <{cpra_fc_threshold} (n={len(low)})")
    ax.step(kmf_high.timeline, prob_high, where="post",
            color=CONFIG["km_colors"][1], linewidth=2.5, alpha=0.8,
            label=f"cPRA-FC ≥{cpra_fc_threshold} (n={len(high)})")

    lr = logrank_test(low["time_to_transplant_days"],
                      high["time_to_transplant_days"],
                      low["transplant_event"],
                      high["transplant_event"])

    def rate_at_2y(time_days, event):
        transplanted = ((time_days <= 730) & (event == 1)).sum()
        with_followup = ((time_days >= 730) | (event == 1)).sum()
        if with_followup == 0:
            return 0
        return transplanted / with_followup * 100

    rate_low = rate_at_2y(low["time_to_transplant_days"], low["transplant_event"])
    rate_high = rate_at_2y(high["time_to_transplant_days"], high["transplant_event"])

    p_text = f"{lr.p_value:.4f}" + ("*" if lr.p_value <= 0.05 else "")
    stats_text = (
        f"Log-rank p = {p_text}\n"
        f"2-year transplant rate:\n{rate_low:.0f}% vs {rate_high:.0f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=CONFIG["annotation_fontsize"], ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    ax.set_xlabel("Days since delisting", fontsize=CONFIG["label_fontsize"], fontweight="bold")
    ax.set_ylabel("Cumulative transplant probability",
                  fontsize=CONFIG["label_fontsize"], fontweight="bold")
    title = (f"{panel_label}. Kaplan-Meier: cPRA ≥{cpra_threshold}%, "
             f"cPRA-FC ≥{cpra_fc_threshold} (N={len(pop)})")
    ax.set_title(title, fontsize=CONFIG["title_fontsize"], fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1095)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patients = load_patients()
    pivot_standard = load_standard_grid()
    pivot_bootstrap = load_bootstrap_grid()

    if pivot_standard is not None and pivot_bootstrap is not None:
        bootstrap_cpra = sorted(pivot_bootstrap.index)
        bootstrap_fc = sorted(pivot_bootstrap.columns)
        pivot_bootstrap = pivot_bootstrap.reindex(
            index=bootstrap_cpra, columns=bootstrap_fc, fill_value=0,
        )
        pivot_standard = pivot_standard.reindex(
            index=bootstrap_cpra, columns=bootstrap_fc, fill_value=1.0,
        )

    fig, axes = plt.subplots(1, 2, figsize=CONFIG["figure_size"])
    create_significance_zones_panel(axes[0], pivot_standard, pivot_bootstrap)
    create_kaplan_meier_panel(
        axes[1], patients,
        CONFIG["operating_point"]["cpra"],
        CONFIG["operating_point"]["cpra_fc"],
        panel_label="B",
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)

    plt.savefig(OUTPUT_DIR / "figure1.png", dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "figure1.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
