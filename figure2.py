"""
figure2.py — Figure 2 of the research letter (3 panels)

Panel A: Detection rate of DSA > 2000 MFI per time interval post-Day 0,
         stratified by DSA category at transplant.
Panel B: Event-free survival (graft loss or death) by DSA category.
Panel C: Forest plot of the multivariate Cox model (static values).

Inputs (data/):
    figure2_patients.csv  (one row per patient)
    figure2_serums.csv    (one row per serum, window [-365, +730] days)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"

CONFIG = {
    "patients_csv": DATA_DIR / "figure2_patients.csv",
    "serums_csv": DATA_DIR / "figure2_serums.csv",
    "time_intervals": [
        (0, 90, "Day 0 to 3 months"),
        (90, 180, "3-6 months"),
        (180, 365, "6-12 months"),
        (365, 730, "12-24 months"),
    ],
    "category_labels": {
        1: "Category 1\nNo DSA >2000 pre-transplant",
        2: "Category 2\nHistorical DSA >2000, cleared at D0",
        3: "Category 3\nPersistent DSA >2000 at D0",
    },
    "dsa_threshold": 2000,
    "max_followup_years": 3,
    "min_patients_per_group": 5,
    "figure_size": (16, 12),
    "colors": ["#2E86AB", "#A23B72", "#F18F01"],
    "dpi": 300,
}


def day0_offset(patient_serums):
    day0 = patient_serums[patient_serums["is_day0"]]
    if len(day0) > 0:
        return int(day0["days_since_transplant"].iloc[0])
    return int(patient_serums.loc[
        patient_serums["days_since_transplant"].abs().idxmin(),
        "days_since_transplant",
    ])


def analyze_dsa_by_intervals(patients, serums):
    results = {}
    serums_by_patient = dict(tuple(serums.groupby("patient_id")))

    for category in sorted(patients["dsa_category"].unique()):
        cat_ids = patients[patients["dsa_category"] == category]["patient_id"].tolist()
        interval_results = {}

        for start_day, end_day, label in CONFIG["time_intervals"]:
            with_dsa, total = 0, 0
            for pid in cat_ids:
                if pid not in serums_by_patient:
                    continue
                s = serums_by_patient[pid]
                d0 = day0_offset(s)
                interval_start = d0 + start_day
                interval_end = d0 + end_day

                in_window = s[
                    (s["days_since_transplant"] >= interval_start)
                    & (s["days_since_transplant"] < interval_end)
                ]
                if len(in_window) == 0:
                    if s["days_since_transplant"].max() < interval_start:
                        continue

                total += 1
                if (in_window["mfi_max"] > CONFIG["dsa_threshold"]).any():
                    with_dsa += 1

            percentage = (with_dsa / total * 100) if total > 0 else 0
            interval_results[label] = {
                "patients_with_dsa": with_dsa,
                "total_patients": total,
                "percentage": percentage,
            }

        results[category] = interval_results

    return results


def create_dsa_histogram_panel(ax, results):
    intervals = [lbl for _, _, lbl in CONFIG["time_intervals"]]
    categories = sorted(results.keys())
    category_short = [f"Category {c}" for c in categories]

    data_for_plot = []
    effectifs_info = []
    for c in categories:
        percentages, effectifs = [], []
        for interval in intervals:
            data = results[c].get(interval, {"patients_with_dsa": 0, "total_patients": 0, "percentage": 0})
            percentages.append(data["percentage"])
            effectifs.append(f"{data['patients_with_dsa']}/{data['total_patients']}")
        data_for_plot.append(percentages)
        effectifs_info.append(effectifs)

    x = np.arange(len(intervals))
    width = 0.25

    for i, (percentages, color, cat_short) in enumerate(zip(data_for_plot, CONFIG["colors"], category_short)):
        bars = ax.bar(x + i * width, percentages, width, label=cat_short,
                      color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        for bar, percentage in zip(bars, percentages):
            if percentage > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{percentage:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    max_pct = max((max(row) for row in data_for_plot), default=10)
    for i, interval in enumerate(intervals):
        y_position = -max_pct * 0.025
        x_position = i + width
        effectifs_text = "\n".join(
            f"{cs}: {effectifs[i]}"
            for cs, effectifs in zip(category_short, effectifs_info)
        ).replace("Category", "Cat.")
        ax.text(x_position, y_position, effectifs_text,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    ax.set_title("A. DSA >2000 MFI Detection by Time Interval",
                 fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Time Interval Post-Day 0", fontweight="bold", fontsize=11)
    ax.set_ylabel("Patients with DSA >2000 MFI (%)", fontweight="bold", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(intervals, fontsize=10)
    ax.legend(fontsize=10, loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-max_pct * 0.20, max_pct * 1.30)


def prepare_survival_data(patients):
    max_followup_days = CONFIG["max_followup_years"] * 365.25
    records = []
    for _, row in patients.iterrows():
        event = row["event_composite"]
        time_days = row["time_composite_days"]
        if pd.isna(event) or pd.isna(time_days):
            continue

        if time_days == 0 and event == 0:
            follow = row.get("time_followup_days")
            if pd.notna(follow) and follow > 0:
                time_days = follow

        if time_days > max_followup_days:
            time_days = max_followup_days
            event = 0

        records.append({
            "patient_id": row["patient_id"],
            "dsa_category": int(row["dsa_category"]),
            "Time_Days": time_days,
            "Event": int(event),
        })
    return pd.DataFrame(records)


def pairwise_logrank(survival_df, valid_categories):
    results = []
    for i, j in [(a, b) for a in range(len(valid_categories))
                 for b in range(len(valid_categories)) if a < b]:
        c1, c2 = valid_categories[i], valid_categories[j]
        g1 = survival_df[survival_df["dsa_category"] == c1]
        g2 = survival_df[survival_df["dsa_category"] == c2]
        try:
            lr = logrank_test(g1["Time_Days"], g2["Time_Days"],
                              g1["Event"], g2["Event"])
            results.append({"c1": c1, "c2": c2, "p_value": lr.p_value})
        except Exception:
            continue
    return results


def create_survival_panel(ax, survival_df):
    valid_categories = sorted([
        c for c in survival_df["dsa_category"].unique()
        if (survival_df["dsa_category"] == c).sum() >= CONFIG["min_patients_per_group"]
    ])
    if len(valid_categories) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    kmf_dict = {}
    for i, c in enumerate(valid_categories):
        group = survival_df[survival_df["dsa_category"] == c]
        events = int(group["Event"].sum())
        total = len(group)
        label = f"Category {c} (n={total}, events={events})"

        kmf = KaplanMeierFitter()
        kmf.fit(group["Time_Days"], event_observed=group["Event"], label=label)
        kmf_dict[f"Category {c}"] = kmf

        ax.step(kmf.timeline, kmf.survival_function_.iloc[:, 0], where="post",
                color=CONFIG["colors"][i], linewidth=2.5, alpha=0.8, label=label)

    ax.set_xlabel("Time since transplant (days)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Event-free survival", fontweight="bold", fontsize=11)
    ax.set_title("B. Event-Free Survival by DSA Category",
                 fontweight="bold", fontsize=13, pad=12)

    pair_results = pairwise_logrank(survival_df, valid_categories)
    if pair_results:
        lines = ["Log-rank tests:"]
        for r in pair_results:
            p = r["p_value"]
            if p < 0.001:
                p_str = "p < 0.001"
            else:
                p_str = f"p = {p:.3f}"
            stars = ""
            if p < 0.05:
                stars = " *"
                if p < 0.01:
                    stars = " **"
                    if p < 0.001:
                        stars = " ***"
            lines.append(f"Category {r['c1']} vs Category {r['c2']}: {p_str}{stars}")
        ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=9, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, CONFIG["max_followup_years"] * 365.25)

    try:
        add_at_risk_counts(*list(kmf_dict.values()), ax=ax,
                           labels=list(kmf_dict.keys()),
                           rows_to_show=["At risk"], ypos=-0.35, fontsize=9)
    except Exception as e:
        print(f"Warning: Could not add at-risk table: {e}")


def create_forest_plot_panel(ax):
    variables = [
        "MFI at transplant Sum\nA/B/DR/DQ (per 1000 MFI)",
        "MFI at transplant Sum\nC/DP (per 1000 MFI)",
        "Peak interval (per 1000 days)",
        "Historical peak (per 1000 MFI)",
        "Age (per year)",
        "Male sex",
        "First transplant",
    ]
    hr = [1.191, 0.757, 1.049, 0.991, 1.032, 0.944, 1.221]
    ci_low = [1.034, 0.311, 0.863, 0.937, 1.015, 0.620, 0.716]
    ci_high = [1.372, 1.841, 1.274, 1.048, 1.049, 1.439, 2.081]
    p_values = [0.015, 0.539, 0.633, 0.753, 0.0002, 0.790, 0.463]

    y_pos = range(len(variables))
    for i in y_pos:
        color = "red" if p_values[i] < 0.05 else "gray"
        ax.plot([ci_low[i], ci_high[i]], [i, i], color=color, linewidth=2)
        ax.plot(hr[i], i, "o", color=color, markersize=6)

    ax.axvline(x=1, color="black", linestyle="--", alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(variables, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Hazard Ratio", fontweight="bold", fontsize=11)
    ax.set_xlim(0.2, 2.5)
    ax.set_title("C. Cox Multivariate Analysis", fontweight="bold", fontsize=13, pad=12)

    for i in y_pos:
        p_text = f"p={p_values[i]:.3f}" if p_values[i] >= 0.001 else "p<0.001"
        ax.text(2.6, i, f"{hr[i]:.2f} ({ci_low[i]:.2f}-{ci_high[i]:.2f}), {p_text}",
                va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patients = pd.read_csv(CONFIG["patients_csv"])
    serums = pd.read_csv(CONFIG["serums_csv"])

    print(f"Patients: {len(patients)}")
    print(f"Sérums  : {len(serums)} ({serums['patient_id'].nunique()} patients)")
    for c in sorted(patients["dsa_category"].unique()):
        print(f"  Category {c}: {(patients['dsa_category'] == c).sum()} patients")

    fig = plt.figure(figsize=CONFIG["figure_size"])
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1],
                          hspace=0.4, wspace=0.3,
                          left=0.06, right=0.94, top=0.95, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 1:3])

    results = analyze_dsa_by_intervals(patients, serums)
    create_dsa_histogram_panel(ax1, results)

    survival_df = prepare_survival_data(patients)
    create_survival_panel(ax2, survival_df)

    create_forest_plot_panel(ax3)

    plt.savefig(OUTPUT_DIR / "figure2.png", dpi=CONFIG["dpi"],
                bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUT_DIR / "figure2.pdf", format="pdf",
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure 2 saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    plt.style.use("default")
    sns.set_context("paper", font_scale=1.0)
    plt.rcParams.update({
        "font.size": 10,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    main()
