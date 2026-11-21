"""Exploratory data analysis helpers. Figures are saved to the given folder."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))


def descriptive_stats(df):
    desc = df.describe().T
    desc["top_pct"] = (desc["freq"] / len(df) * 100).round(1)
    return desc.sort_values("top_pct", ascending=False)


def rare_categories(df, target="class", threshold=0.01):
    """Categorical equivalent of outliers: levels seen in < threshold of rows."""
    rows = []
    for col in df.columns.drop(target):
        freq = df[col].value_counts(normalize=True)
        for level, share in freq[freq < threshold].items():
            mask = df[col] == level
            rows.append({"Feature": col, "Category": level, "Count": int(mask.sum()),
                         "Share (%)": round(share * 100, 2),
                         "% Poisonous": round((df.loc[mask, target] == "p").mean() * 100, 1)})
    return pd.DataFrame(rows).sort_values("Count")


def target_association(df, target="class", exclude=("veil-type",)):
    rows = [[c, cramers_v(df[c], df[target])]
            for c in df.columns if c != target and c not in exclude]
    return (pd.DataFrame(rows, columns=["Feature", "Cramers_V"])
              .sort_values("Cramers_V", ascending=False))


def feature_association_matrix(df, target="class", exclude=("veil-type",)):
    feats = [c for c in df.columns if c != target and c not in exclude]
    m = pd.DataFrame(index=feats, columns=feats, dtype=float)
    for a in feats:
        for b in feats:
            m.loc[a, b] = 1.0 if a == b else cramers_v(df[a], df[b])
    return m


def save_eda_figures(df, assoc_target, assoc_matrix, out_dir="figures"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    ax = assoc_target.plot(x="Feature", y="Cramers_V", kind="bar", legend=False, figsize=(12, 6))
    ax.set_title("Cramér's V Association Between Features and Mushroom Class")
    plt.tight_layout(); plt.savefig(out / "cramers_v_target.png", dpi=120); plt.close()

    top5 = assoc_target["Feature"].head(5).tolist()
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, col in zip(axes, top5):
        pd.crosstab(df[col], df["class"]).plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(col); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(out / "top5_features_by_class.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(assoc_matrix.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(assoc_matrix))); ax.set_xticklabels(assoc_matrix.columns, rotation=90)
    ax.set_yticks(range(len(assoc_matrix))); ax.set_yticklabels(assoc_matrix.index)
    fig.colorbar(im, ax=ax, label="Cramér's V")
    ax.set_title("Cramér's V Between Features")
    plt.tight_layout(); plt.savefig(out / "cramers_v_heatmap.png", dpi=120); plt.close()
