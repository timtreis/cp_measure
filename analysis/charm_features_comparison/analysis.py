"""Information-content comparison: cp_measure vs CharmFeatures on BBBC022 sites.

Samples: 72 sites (4 plates x 6 wells x 3 sites).
Labels: DMSO (wells a13, e01, g19), mitoxantrone (c02), simvastatin (c08),
etoposide (o08).
"""
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score

S = "/tmp/claude-0/-home-user-cp-measure/b903f149-ecdd-5071-9248-1b294ef3b859/scratchpad"
CHAN = {1: "DNA", 2: "ER", 3: "RNA", 4: "AGP", 5: "Mito"}
WELL2CPD = {"a13": "DMSO", "e01": "DMSO", "g19": "DMSO",
            "c02": "mitoxantrone", "c08": "simvastatin", "o08": "etoposide"}

# ---------- load charm ----------
z = np.load(f"{S}/charm_features_sq.npz", allow_pickle=True)
samples = [str(s) for s in z["samples"]]
F = np.asarray(z["features"], dtype=float)
import json
names = json.load(open(f"{S}/charm_names.json"))
rows = {}
for s, fv in zip(samples, F):
    m = re.search(r"(\d+)_([a-z]\d+)_s(\d)_w(\d)", s)
    site = f"{m.group(1)}_{m.group(2)}_s{m.group(3)}"
    rows.setdefault(site, {})
    ch = CHAN[int(m.group(4))]
    for n, v in zip(names, fv):
        rows[site][f"{n}__{ch}"] = v
charm = pd.DataFrame(rows).T.sort_index()
print("charm:", charm.shape)

# ---------- load cp_measure ----------
cp = pd.read_parquet(f"{S}/cp_features_sq.parquet").sort_index()
print("cp_measure:", cp.shape)

idx = charm.index.intersection(cp.index)
charm, cp = charm.loc[idx], cp.loc[idx]
meta = pd.DataFrame({
    "plate": [s.split("_")[0] for s in idx],
    "well": [s.split("_")[1] for s in idx],
    "cpd": [WELL2CPD[s.split("_")[1]] for s in idx],
}, index=idx)
print(meta.groupby("cpd").size().to_dict())


def clean(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    frac = df.notna().mean(axis=0)
    df = df.loc[:, frac >= 0.9]
    df = df.fillna(df.median(axis=0))
    sd = df.std(axis=0)
    return df.loc[:, sd > 1e-12]


def robust_z(df, meta):
    """Per-plate robust z vs DMSO."""
    out = []
    for p, sub in df.groupby(meta["plate"]):
        ctrl = sub.loc[meta.loc[sub.index, "cpd"] == "DMSO"]
        med = ctrl.median(axis=0)
        mad = (ctrl - med).abs().median(axis=0) * 1.4826
        mad = mad.replace(0, np.nan).fillna(ctrl.std(axis=0).replace(0, 1))
        mad = mad.replace(0, 1)
        out.append((sub - med) / mad)
    z = pd.concat(out).loc[df.index]
    return clean(z)


charm_c, cp_c = clean(charm), clean(cp)
charm_z, cp_z = robust_z(charm_c, meta), robust_z(cp_c, meta)
print("after clean/normalize:", charm_z.shape, cp_z.shape)

groups = meta["plate"].values
logo = LeaveOneGroupOut()


def r2_cv(X, Y, n_pc=40):
    """Leave-one-plate-out R2 for predicting each column of Y from X-PCs."""
    Xs = StandardScaler().fit_transform(X)
    r2 = np.zeros(Y.shape[1])
    preds = np.zeros(Y.shape)
    for tr, te in logo.split(Xs, groups=groups):
        pca = PCA(n_components=min(n_pc, len(tr) - 1)).fit(Xs[tr])
        Ztr, Zte = pca.transform(Xs[tr]), pca.transform(Xs[te])
        model = Ridge(alpha=10.0).fit(Ztr, Y.values[tr])
        preds[te] = model.predict(Zte)
    ss_res = ((Y.values - preds) ** 2).sum(axis=0)
    ss_tot = ((Y.values - Y.values.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


print("\n=== A) predict charm features from cp_measure features ===")
r2_charm = r2_cv(cp_z, charm_z)
res = pd.DataFrame({"r2": r2_charm}, index=charm_z.columns)
res["family"] = [c.split(" (")[0] for c in res.index]
res["transform"] = [re.search(r"\((.*)\) \[", c).group(1).strip() or "raw"
                    for c in res.index]
res["channel"] = [c.split("__")[-1] for c in res.index]
res.to_parquet(f"{S}/r2_charm_from_cp.parquet")
print("median R2:", np.median(r2_charm).round(3),
      "| frac R2<0.2:", (r2_charm < 0.2).mean().round(3),
      "| frac R2>0.8:", (r2_charm > 0.8).mean().round(3))
print("\nby family (median R2, worst-explained first):")
fam = res.groupby("family")["r2"].median().sort_values()
print(fam.round(3).to_string())
print("\nby transform (median R2):")
print(res.groupby("transform")["r2"].median().sort_values().round(3).to_string())

print("\n=== B) predict cp_measure features from charm features ===")
r2_cp = r2_cv(charm_z, cp_z)
res_b = pd.DataFrame({"r2": r2_cp}, index=cp_z.columns)


def cp_family(c):
    c2 = c.split("::")[-1]
    base = c2.split("__")[0]
    base = re.sub(r"_?\d+_?\d*$", "", base)
    for k in ["RadialDistribution", "Granularity", "Intensity", "Zernike",
              "Correlation", "Location", "InfoMeas"]:
        if base.startswith(k):
            return k
    tex = ["AngularSecondMoment", "Contrast", "Variance", "InverseDifferenceMoment",
           "SumAverage", "SumVariance", "SumEntropy", "Entropy",
           "DifferenceVariance", "DifferenceEntropy"]
    if any(base.startswith(t) for t in tex):
        return "Texture(Haralick)"
    return "SizeShape"


res_b["family"] = [cp_family(c) for c in res_b.index]
res_b.to_parquet(f"{S}/r2_cp_from_charm.parquet")
print("median R2:", np.median(r2_cp).round(3),
      "| frac R2<0.2:", (r2_cp < 0.2).mean().round(3))
print(res_b.groupby("family")["r2"].median().sort_values().round(3).to_string())

print("\n=== C) classification: compound identity, leave-one-plate-out ===")
y = meta["cpd"].values


def clf_acc(X, n_pc=30):
    Xs = StandardScaler().fit_transform(X)
    accs = []
    for tr, te in logo.split(Xs, groups=groups):
        pca = PCA(n_components=min(n_pc, len(tr) - 1)).fit(Xs[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(pca.transform(Xs[tr]), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(pca.transform(Xs[te]))))
    return np.mean(accs)


both = pd.concat([cp_z, charm_z], axis=1)
print("cp_measure only :", round(clf_acc(cp_z), 3))
print("charm only      :", round(clf_acc(charm_z), 3))
print("combined        :", round(clf_acc(both), 3))

print("\n=== D) replicate retrieval (NN same-compound, cosine, cross-plate) ===")


def nn_score(X):
    Xs = StandardScaler().fit_transform(X)
    Xs = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
    sim = Xs @ Xs.T
    correct = 0
    n = 0
    for i in range(len(idx)):
        mask = (groups != groups[i])
        j = np.argmax(np.where(mask, sim[i], -np.inf))
        correct += (y[j] == y[i])
        n += 1
    return correct / n


print("cp_measure only :", round(nn_score(cp_z), 3))
print("charm only      :", round(nn_score(charm_z), 3))
print("combined        :", round(nn_score(both), 3))

print("\n=== E) least-explained charm features (top 25) ===")
worst = res.sort_values("r2").head(25)
print(worst[["r2", "family", "transform", "channel"]].to_string())
