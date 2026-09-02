"""Segment + featurize Cell Painting sites with cp_measure.

Per site: nuclei from DNA channel (Otsu + distance-transform watershed),
cells = expand_labels(nuclei, 25 px). Features via cp_measure.featurizer,
median-aggregated per site.
"""
import sys, time, glob, re, warnings
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed, expand_labels
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from cp_measure.featurizer import make_featurizer_config, featurize

CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]  # w1..w5

def segment(dna):
    sm = gaussian(dna.astype(np.float32), sigma=2)
    thr = threshold_otsu(sm)
    fg = sm > thr
    fg = remove_small_objects(fg, 100)
    dist = ndi.distance_transform_edt(fg)
    peaks = peak_local_max(dist, min_distance=10, labels=fg)
    seeds = np.zeros(dna.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, 1):
        seeds[r, c] = i
    nuclei = watershed(-dist, seeds, mask=fg)
    nuclei = remove_small_objects(nuclei, 100)
    cells = expand_labels(nuclei, distance=25)
    # keep only labels present in nuclei
    return nuclei.astype(np.int32), cells.astype(np.int32)

def process_site(imgdir, plate, well, site, config):
    chans = []
    for w in range(1, 6):
        chans.append(tifffile.imread(f"{imgdir}/{plate}_{well}_s{site}_w{w}.tif"))
    image = np.stack(chans).astype(np.float32) / 65535.0
    nuclei, cells = segment(image[0])
    if nuclei.max() < 5:
        return None
    masks = np.stack([nuclei, cells])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, cols, rows = featurize(image, masks, config,
                                     image_id=f"{plate}_{well}_s{site}")
    df = pd.DataFrame(data, columns=cols)
    df["__obj"] = [r[1] for r in rows]
    parts = []
    for obj, sub in df.groupby("__obj"):
        med = sub.drop(columns="__obj").median(axis=0, skipna=True)
        med.index = [f"{obj}::{c}" for c in med.index]
        parts.append(med)
    med = pd.concat(parts)
    med["n_objects"] = float(len(np.unique(nuclei)) - 1)
    return med

if __name__ == "__main__":
    S = "/tmp/claude-0/-home-user-cp-measure/b903f149-ecdd-5071-9248-1b294ef3b859/scratchpad"
    config = make_featurizer_config(CHANNELS, objects=["nuclei", "cells"])
    sites = sorted({tuple(re.match(r"(\d+)_(\w+)_s(\d)_w1", f.split("/")[-1]).groups())
                    for f in glob.glob(f"{S}/images_sq/*_w1.tif")})
    which = sys.argv[1:] if len(sys.argv) > 1 else None
    out = {}
    t0 = time.time()
    for i, (p, w, s) in enumerate(sites):
        if which and f"{p}_{w}_s{s}" not in which:
            continue
        med = process_site(f"{S}/images_sq", p, w, s, config)
        if med is not None:
            out[f"{p}_{w}_s{s}"] = med
        print(f"[{i+1}/{len(sites)}] {p}_{w}_s{s} done, {time.time()-t0:.0f}s elapsed", flush=True)
    df = pd.DataFrame(out).T
    df.to_parquet(f"{S}/cp_features_sq.parquet")
    print("saved", df.shape)
