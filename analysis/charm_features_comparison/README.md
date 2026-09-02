# CharmFeatures (WND-CHARM) vs cp_measure: coverage and information content

Question: what does [CharmFeatures](https://gitlab.com/iggman/charm-features)
(a ctypes wrapper around WND-CHARM's libcharm) measure that cp_measure does
not, does that difference carry usable information on real Cell Painting
data, and could the math simply be ported?

## 1. Feature-space coverage

CharmFeatures computes 2895 features per grayscale image (or tile). The set
is a cross product of 15 feature algorithms and 12 image domains (raw plus
11 transform chains built from Fourier, Wavelet (Symlet-5), Chebyshev, and
Prewitt-edge transforms):

| Algorithm (features) | In cp_measure? |
| --- | --- |
| Haralick Textures (26) | Yes (per object, raw domain only) |
| Zernike Coefficients (72) | Related (shape Zernike on masks, radial Zernike per object); charm's is intensity Zernike of the whole image |
| Pixel Intensity Statistics (5) | Yes, far richer per object |
| Edge Features (28, Prewitt magnitude/direction stats) | Partially (edge intensity per object; no gradient-direction features) |
| Otsu / Inverse-Otsu Object Features (34+34) | Superseded by real segmentation |
| Multiscale Histograms (24) | No |
| Tamura Textures (6) | No |
| Comb Moments (48, directional first-4-moment profiles) | No |
| Radon Coefficients (12) | No |
| Chebyshev Coefficients (32) | No |
| Chebyshev-Fourier Coefficients (32) | No |
| Fractal Features (20) | No |
| Gabor Textures (7) | No |
| Gini Coefficient (1) | No |
| Color Histogram | N/A (fluorescence) |

The transform-domain idea (any algorithm computed on FFT/wavelet/Chebyshev/
edge-transformed images, including 2-deep chains) has no counterpart in
cp_measure at all; it accounts for ~87% of the 2895 features.

Conversely, charm is single-channel, whole-image and segmentation-free, so
it has no counterpart for cp_measure's per-object geometry, intensity,
granularity spectrum, radial distributions, colocalization, or neighbor
features.

## 2. Empirical information content (BBBC022 / cpg0012)

Data: 4 replicate plates (24277, 24296, 24308, 24309) of the BBBC022 Cell
Painting bioactives set (U2OS, 5 channels), 6 wells per plate (3x DMSO,
mitoxantrone, simvastatin, etoposide), 3 sites per well, center-cropped to
520x520 (see section 4 for why square). Per site: cp_measure upstream main
via `featurizer` on nuclei (Otsu + watershed on DNA) and cells
(expand_labels 25 px), median-aggregated -> 2071 features; CharmFeatures
per channel -> 14475 features. Profiles per-plate robust-z scored against
DMSO. n = 71 sites.

Predicting each feature set from the other (ridge on 40 PCs,
leave-one-plate-out CV):

- charm from cp_measure: median R2 = -0.10; 78% of charm features have
  R2 < 0.2; only 0.2% exceed 0.8.
- cp_measure from charm: median R2 = 0.16. Intensity (0.39), InfoMeas
  (0.34), Correlation (0.28) and Haralick (0.27) are partially captured;
  Granularity (-0.01), SizeShape (0.05), Zernike (0.04),
  RadialDistribution (0.09), Location (-0.11) are essentially invisible to
  charm.

Low predictability of charm features is largely noise, not novelty. Feature
reliability (mean cross-plate Spearman of per-well medians): charm median
0.31 vs cp_measure 0.49; fraction with reliability > 0.5: 26% vs 46%.
Predictability rises monotonically with reliability (median R2 by
reliability bin: -0.26, -0.05, 0.16, 0.35). Still, among the 3715 reliable
charm features, 81% have R2 < 0.5 from cp_measure, i.e. there IS
reproducible charm variance that cp_measure does not linearly encode,
concentrated in transform-domain Haralick, image Zernike, Comb Moments,
Fractal, Multiscale Histograms and Chebyshev(-Fourier) coefficients.

Does that variance help downstream? On this dataset, no:

| task (leave-one-plate-out) | cp_measure | charm | combined |
| --- | --- | --- | --- |
| 4-class compound classification | 0.93 | 0.85 | 0.90 |
| cross-plate NN replicate retrieval | 0.96 | 0.85 | 0.87 |

Adding charm never beats cp_measure alone and slightly dilutes it.
Caveats: only 3 compounds + DMSO, cp_measure is near ceiling, one cell
line, image-level profiles, linear probes. A larger compound panel could
still surface complementary signal; this experiment bounds it as small.

## 3. Can the math be ported? Yes.

The algorithms are small, self-contained numeric routines (roughly 100-350
lines of C++ each, no external state). `validate_ports.py` ports four of
them (Gini, Fractal, Multiscale Histograms, Pixel Intensity Statistics) in
~15 lines of numpy each; they match libcharm to <= 2e-6 relative error
(float32 input precision). The transform stack is equally portable
(numpy.fft, pywt sym5, numpy.polynomial.chebyshev; skimage has radon and
gabor). If porting, the sensible target is the genuinely uncovered math
(Tamura, Gabor, fractal, Gini, multiscale histograms, comb moments,
Chebyshev/Chebyshev-Fourier/Radon coefficients, plus the transform-domain
trick as an optional image pre-transform), applied per object crop to fit
cp_measure's mask-based paradigm, rather than wrapping libcharm.

## 4. Bug found in CharmFeatures (upstream-relevant)

`CharmFeatures/__init__.py` passes `width, height = ndarray.shape` (i.e.
swapped) into libcharm, which maps the buffer as a row-major Eigen matrix
of shape (height, width). For square images this is a harmless transpose;
for non-square images every spatially aware feature (Haralick, edges,
fractal, Zernike, wavelets, ...) is computed on a diagonally scrambled
pixel array. Verified numerically: our fractal port matches libcharm to
1e-8 on non-square input only when the image is deliberately scrambled the
same way (`A.ravel().reshape(w, h).T`), and matches the plain image once
inputs are square. Intensity-histogram features (Gini, multiscale
histograms, pixel stats) are unaffected. All analyses here therefore use
520x520 crops. Worth reporting to the CharmFeatures maintainer.

## Files

- `cp_pipeline.py`: segmentation + cp_measure featurization per site.
- `analysis.py`: cross-prediction R2, classification, retrieval.
- `analysis2.py`: reliability analysis (signal vs noise decomposition).
- `validate_ports.py`: numpy ports of four charm algorithms + validation.

Scripts expect the BBBC022 crops and CharmFeatures outputs in a scratch
directory (paths at the top of each file); they are kept for provenance,
not as a maintained pipeline.
