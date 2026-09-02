"""Numpy ports of four WND-CHARM feature algorithms, validated against libcharm.

Run after CharmFeatures has produced a sidecar .npz for a SQUARE tiff
(see README: the CharmFeatures ctypes wrapper scrambles non-square inputs).

Usage: python validate_ports.py <image.tif> <charm_names.json>
"""
import sys
import json
import numpy as np
import tifffile


def gini(img):
    """Gini coefficient of positive pixels (Abraham et al. 2003)."""
    v = np.sort(img.ravel()[img.ravel() > 0])
    n = len(v)
    if n <= 1:
        return 0.0
    mean = v.mean()
    g = np.sum((2 * np.arange(1, n + 1) - n - 1) * v)
    return g / (mean * n * (n - 1))


def fractal(img, bins=20):
    """Brownian fractal signature: mean |lag-k| difference, 20 lags (Wu et al. 1992)."""
    h, w = img.shape
    K = min(h, w) // 5
    step = max(K // bins, 1)
    out = []
    for k in range(1, K, step):
        s = np.abs(img[:-k, :] - img[k:, :]).sum() + np.abs(img[:, :-k] - img[:, k:]).sum()
        out.append(s / (w * (w - k) + h * (h - k)))
    return np.array(out[:bins])


def multiscale_histograms(img):
    """Histograms with 3/5/7/9 bins over [min, max], jointly max-normalized."""
    mn, mx = img.min(), img.max()
    out = []
    for nb in (3, 5, 7, 9):
        scale = nb / (mx - mn) if mx > mn else 0
        b = np.minimum(((img.ravel() - mn) * scale).astype(np.int64), nb - 1)
        out.append(np.bincount(b, minlength=nb).astype(float))
    out = np.concatenate(out)
    return out / out.max()


def pixel_intensity_statistics(img):
    p = img.ravel()
    return np.array([p.mean(), np.median(p), p.std(ddof=0), p.min(), p.max()])


if __name__ == "__main__":
    tif, names_json = sys.argv[1], sys.argv[2]
    names = json.load(open(names_json))
    npz = np.load(tif.replace(".tif", ".npz"))
    key = [k for k in npz.files if not k.startswith("_")][0]
    fv = npz[key].astype(np.float64)
    # libcharm receives float32 pixels; match that quantization
    img = tifffile.imread(tif).astype(np.float32).astype(np.float64)
    assert img.shape[0] == img.shape[1], "use a square image (see README)"

    def idxs(prefix):
        return [i for i, n in enumerate(names) if n.startswith(prefix + " () [")]

    ports = {
        "Gini Coefficient": np.atleast_1d(gini(img)),
        "Fractal Features": fractal(img),
        "Multiscale Histograms": multiscale_histograms(img),
        "Pixel Intensity Statistics": pixel_intensity_statistics(img),
    }
    for name, mine in ports.items():
        ref = fv[idxs(name)]
        err = np.max(np.abs(mine - ref) / (np.abs(ref) + 1e-9))
        print(f"{name:28s} max rel err vs libcharm: {err:.2e}")
