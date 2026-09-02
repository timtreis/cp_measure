"""Follow-up: is unexplained charm variance signal or noise?

Reliability = Spearman correlation of per-well median profiles between
plate pairs (feature-wise, across the 6 wells), averaged over pairs.
A feature must be reproducible across replicate plates to carry information.
Then relate reliability to predictability-from-cp (R2 from analysis.py).
"""
import re
import numpy as np
import pandas as pd
import json

S = "/tmp/claude-0/-home-user-cp-measure/b903f149-ecdd-5071-9248-1b294ef3b859/scratchpad"
CHAN = {1: "DNA", 2: "ER", 3: "RNA", 4: "AGP", 5: "Mito"}

z = np.load(f"{S}/charm_features_sq.npz", allow_pickle=True)
samples = [str(s) for s in z["samples"]]
F = np.asarray(z["features"], dtype=float)
names = json.load(open(f"{S}/charm_names.json"))
rows = {}
for s, fv in zip(samples, F):
    m = re.search(r"(\d+)_([a-z]\d+)_s(\d)_w(\d)", s)
    site = f"{m.group(1)}_{m.group(2)}_s{m.group(3)}"
    rows.setdefault(site, {})
    for n, v in zip(names, fv):
        rows[site][f"{n}__{CHAN[int(m.group(4))]}"] = v
charm = pd.DataFrame(rows).T.sort_index()
cp = pd.read_parquet(f"{S}/cp_features_sq.parquet").sort_index()
idx = charm.index.intersection(cp.index)
charm, cp = charm.loc[idx], cp.loc[idx]

plate = pd.Series([s.split("_")[0] for s in idx], index=idx)
well = pd.Series([s.split("_")[1] for s in idx], index=idx)


def reliability(df):
    """Feature-wise mean cross-plate Spearman of per-well medians."""
    wellmed = df.groupby([plate, well]).median()  # (plate, well) x features
    plates = wellmed.index.get_level_values(0).unique()
    # rank across wells within plate, then correlate between plates
    cors = []
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            a = wellmed.loc[plates[i]]
            b = wellmed.loc[plates[j]]
            common = a.index.intersection(b.index)
            ar = a.loc[common].rank()
            br = b.loc[common].rank()
            c = ((ar - ar.mean()) * (br - br.mean())).sum() / (
                np.sqrt(((ar - ar.mean()) ** 2).sum() * ((br - br.mean()) ** 2).sum()))
            cors.append(c)
    return pd.concat(cors, axis=1).mean(axis=1)


rel_charm = reliability(charm)
rel_cp = reliability(cp)
print("median cross-plate reliability: charm",
      round(rel_charm.median(), 3), "| cp_measure", round(rel_cp.median(), 3))
print("frac reliability > 0.5: charm", round((rel_charm > 0.5).mean(), 3),
      "| cp_measure", round((rel_cp > 0.5).mean(), 3))

r2 = pd.read_parquet(f"{S}/r2_charm_from_cp.parquet")
common = r2.index.intersection(rel_charm.index)
r2 = r2.loc[common]
rel = rel_charm.loc[common]

bins = pd.cut(rel, [-1, 0.2, 0.5, 0.8, 1.0])
summ = pd.DataFrame({"n": r2.groupby(bins).size(),
                     "median_R2_from_cp": r2["r2"].groupby(bins).median().round(3)})
print("\ncharm features grouped by their own cross-plate reliability:")
print(summ.to_string())

reliable = rel > 0.5
print(f"\nreliable charm features (rel>0.5): {reliable.sum()}")
print("of these, median R2 from cp_measure:",
      round(r2.loc[reliable[reliable].index, "r2"].median(), 3))
print("of these, frac with R2 < 0.5 (candidate novel info):",
      round((r2.loc[reliable[reliable].index, "r2"] < 0.5).mean(), 3))

novel = r2.loc[reliable[reliable].index]
novel = novel[novel["r2"] < 0.5]
print("\nreliable-but-unexplained charm features by family:")
print(novel.groupby("family").size().sort_values(ascending=False).to_string())
print("\nby transform:")
print(novel.groupby("transform").size().sort_values(ascending=False).to_string())
novel.sort_values("r2").to_parquet(f"{S}/novel_charm_features.parquet")

# top reliable-novel features with their reliability
novel2 = novel.join(rel.rename("reliability"))
print("\ntop 20 reliable & least explained:")
print(novel2.sort_values("reliability", ascending=False)
      .head(20)[["reliability", "r2", "family", "transform", "channel"]]
      .round(3).to_string())
