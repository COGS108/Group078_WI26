"""
CA County Linear Regression (2018–2023) — Outliers Removed
============================================================
Outcome (Y): overdose deaths per 100K population
Predictors (X): personal_income, labor_force, fmr_2br, unemployment_rate

Outlier removal: IQR method (1.5× IQR) applied independently to both
X and Y for each predictor graph (per-predictor cleaning).

Usage:
    pip install pandas scikit-learn matplotlib seaborn
    python ca_linear_regression_v3.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── 1. Load & prep ────────────────────────────────────────────────────────────

df = pd.read_csv("CA_merged_2018_2023.csv")
df["Year"] = df["Year"].astype(int)
df["overdose_per_100k"] = df["total_overdose_deaths"] / df["population"] * 100_000
df = df.dropna(subset=["overdose_per_100k"])

PREDICTORS = {
    "Personal Income ($)":      "personal_income",
    "Labor Force":               "labor_force",
    "Fair Market Rent 2BR ($)": "fmr_2br",
    "Unemployment Rate (%)":    "unemployment_rate",
}

OUTCOME_LABEL = "Overdose Deaths per 100K"
OUTCOME_COL   = "overdose_per_100k"


# ── 2. Helpers ────────────────────────────────────────────────────────────────

def remove_iqr_outliers(data, cols):
    """Remove rows where any of `cols` fall outside 1.5×IQR fences."""
    mask = pd.Series(True, index=data.index)
    for col in cols:
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        iqr = q3 - q1
        mask &= (data[col] >= q1 - 1.5 * iqr) & (data[col] <= q3 + 1.5 * iqr)
    return data[mask]

def fit_regression(X, y):
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    return model, preds, r2_score(y, preds)


# ── 3. Pooled regressions (with and without outliers) ─────────────────────────

print("\n" + "="*60)
print(f"POOLED REGRESSIONS — Y = {OUTCOME_LABEL}")
print(f"IQR outlier removal applied per predictor (both X and Y)")
print("="*60)

pooled_results   = {}   # cleaned
pooled_raw       = {}   # original (for comparison)

for label, col in PREDICTORS.items():
    sub_raw   = df.dropna(subset=[col])
    sub_clean = remove_iqr_outliers(sub_raw, [col, OUTCOME_COL])

    n_removed = len(sub_raw) - len(sub_clean)

    for tag, sub, store in [("raw", sub_raw, pooled_raw),
                             ("clean", sub_clean, pooled_results)]:
        X = sub[[col]].values
        y = sub[OUTCOME_COL].values
        model, _, r2 = fit_regression(X, y)
        store[label] = {
            "col": col, "slope": model.coef_[0],
            "intercept": model.intercept_, "r2": r2,
            "model": model, "sub": sub,
        }

    print(f"\n{label}")
    print(f"  n (raw / clean) : {len(sub_raw)} / {len(sub_clean)}  "
          f"({n_removed} outliers removed)")
    print(f"  R² before       : {pooled_raw[label]['r2']:.4f}")
    print(f"  R² after        : {pooled_results[label]['r2']:.4f}")
    direction = "↑ positive" if pooled_results[label]['slope'] > 0 else "↓ negative"
    print(f"  Direction       : {direction}")


# ── 4. Per-county regressions (cleaned data) ──────────────────────────────────

print("\n" + "="*60)
print("PER-COUNTY REGRESSIONS (cleaned data)")
print("="*60)

county_results = {label: [] for label in PREDICTORS}

for label, col in PREDICTORS.items():
    sub_clean = pooled_results[label]["sub"]
    for county, grp in sub_clean.groupby("county"):
        grp = grp.sort_values("Year")
        if len(grp) < 3:
            continue
        X = grp[[col]].values
        y = grp[OUTCOME_COL].values
        model, _, r2 = fit_regression(X, y)
        county_results[label].append({
            "county": county, "slope": model.coef_[0],
            "intercept": model.intercept_, "r2": r2,
        })

for label, rows in county_results.items():
    cdf = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print(f"\n── {label}  (top 5 counties by R²) ──")
    print(cdf[["county", "slope", "r2"]].head(5).to_string(index=False))


# ── 5. Regression scatter plots ───────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"CA County: {OUTCOME_LABEL} vs. Predictors\n"
    f"(Outliers removed via IQR · per-predictor cleaning · 2018–2023)",
    fontsize=13, fontweight="bold"
)

scatter_ref = None
for ax, (label, col) in zip(axes.flat, PREDICTORS.items()):
    raw   = pooled_raw[label]
    clean = pooled_results[label]
    sub   = clean["sub"]

    # 5th–95th percentile axis limits for zoom (on cleaned data)
    x_lo, x_hi = sub[col].quantile(0.05), sub[col].quantile(0.95)
    y_lo, y_hi = sub[OUTCOME_COL].quantile(0.05), sub[OUTCOME_COL].quantile(0.95)
    x_pad = (x_hi - x_lo) * 0.04
    y_pad = (y_hi - y_lo) * 0.04

    # Only plot points that fall within the zoomed window
    in_view = (
        sub[col].between(x_lo - x_pad, x_hi + x_pad) &
        sub[OUTCOME_COL].between(y_lo - y_pad, y_hi + y_pad)
    )
    sub_view = sub[in_view]

    scatter_ref = ax.scatter(
        sub_view[col], sub_view[OUTCOME_COL],
        c=sub_view["Year"], cmap="coolwarm", alpha=0.65, s=26, linewidths=0,
        label=f"n={len(sub_view)} observations", zorder=2
    )

    # Regression line clipped to zoom window
    x_range = np.linspace(x_lo - x_pad, x_hi + x_pad, 200).reshape(-1, 1)
    ax.plot(x_range, clean["model"].predict(x_range),
            color="crimson", linewidth=2.2,
            label=f"Fit  R²={clean['r2']:.4f}", zorder=3)

    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel(OUTCOME_LABEL, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=7.5)

plt.colorbar(scatter_ref, ax=axes.flat[-1], label="Year", shrink=0.8)
plt.tight_layout()
plt.savefig("ca_regression_plots_clean.png", dpi=150, bbox_inches="tight")
print("\n✅  Plot saved → ca_regression_plots_clean.png")


# ── 6. Exploratory trend plots (cleaned) ──────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"CA County: {OUTCOME_LABEL} vs. Predictors — County Traces (Outliers Removed)",
    fontsize=13, fontweight="bold"
)

for ax, (label, col) in zip(axes.flat, PREDICTORS.items()):
    sub = pooled_results[label]["sub"]

    for county, grp in sub.groupby("county"):
        grp = grp.sort_values(col)
        ax.plot(grp[col], grp[OUTCOME_COL],
                color="gray", alpha=0.2, linewidth=0.8)

    x_range = np.linspace(sub[col].min(), sub[col].max(), 200).reshape(-1, 1)
    res = pooled_results[label]
    ax.plot(x_range, res["model"].predict(x_range),
            color="navy", linewidth=2.2,
            label=f"Pooled trend  R²={res['r2']:.4f}")

    # 5th–95th percentile zoom
    x_lo, x_hi = sub[col].quantile(0.05), sub[col].quantile(0.95)
    y_lo, y_hi = sub[OUTCOME_COL].quantile(0.05), sub[OUTCOME_COL].quantile(0.95)
    x_pad = (x_hi - x_lo) * 0.04
    y_pad = (y_hi - y_lo) * 0.04
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel(OUTCOME_LABEL, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ca_exploratory_trends_clean.png", dpi=150, bbox_inches="tight")
print("✅  EDA plot saved → ca_exploratory_trends_clean.png")


# ── 7. Export cleaned per-county results ──────────────────────────────────────

all_rows = []
for label, rows in county_results.items():
    for r in rows:
        all_rows.append({"parameter": label, **r})

pd.DataFrame(all_rows)[["parameter","county","slope","intercept","r2"]]\
    .to_csv("ca_regression_per_county_clean.csv", index=False)
print("✅  Per-county results saved → ca_regression_per_county_clean.csv")