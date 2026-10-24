"""Optical depth of a lens population for a source at z_s, as a histogram in image
separation (and optionally time delay):

    tau = sum over z_l in ZL of dV/dz dz  sum_i n_i (1 + z_l)^-nu  pi beta_r,i^2,

with non-evolving comoving densities n_i for nu = 0, and every counted source sample
(lensing.images) carrying an equal share of its lens's cross-section.
"""
import numpy as np

from .cosmology import C_KMS, DAY, ZS, dt_distance_m, dvdz, sigma_crit
from .lensing import images

ZL = np.linspace(0.05, 1.95, 39)
SEP_BINS = np.logspace(-2, np.log10(20), 61)                   # arcsec
SEP_E = np.logspace(-1.3, 1.0, 47)                             # coarser bins for (sep, delay)
DT_E = np.logspace(-1.5, 3.5, 51)                              # days


def optical_depth(rows, w, nbeta=8, qmin=None, nu=0.0, bins=SEP_BINS, dt_bins=None,
                  per_row=False, zs=ZS):
    """Histogram of tau in separation, or in (separation, delay [d]) if dt_bins is given.

    rows: enclosed-mass rows on lensing.R3; w: comoving density of each row [Mpc^-3].
    per_row: return one histogram per row instead of their sum.
    """
    w = np.asarray(w, float)
    shape = (len(bins) - 1,) if dt_bins is None else (len(bins) - 1, len(dt_bins) - 1)
    out = np.zeros(((len(rows),) if per_row else ()) + shape)
    dz = ZL[1] - ZL[0]
    for zl in ZL:
        scrit, dl = sigma_crit(zl, zs)
        beta_r, sep, dphi, _ = images(rows, dl, scrit, nbeta, qmin)
        wt = np.broadcast_to((w * (1 + zl) ** (-nu) * dvdz(zl) * dz * np.pi * beta_r**2 / nbeta)[:, None],
                             sep.shape)
        ok = np.isfinite(sep)
        if not ok.any():
            continue
        coords = [sep[ok]]
        edges = [bins]
        if dt_bins is not None:
            coords.append((dt_distance_m(zl, zs) / (C_KMS * 1e3) * dphi / DAY)[ok])
            edges.append(dt_bins)
        if not per_row:
            out += np.histogramdd(coords, edges, weights=wt[ok])[0]
            continue
        idx = [np.searchsorted(e, x, side="right") - 1 for x, e in zip(coords, edges)]
        for x, e, k in zip(coords, edges, idx):
            k[x == e[-1]] = len(e) - 2                           # right edge is closed, as in np.histogram
        inside = np.all([(k >= 0) & (k < len(e) - 1) for k, e in zip(idx, edges)], axis=0)
        rowi = np.nonzero(ok)[0][inside]
        np.add.at(out, (rowi, *[k[inside] for k in idx]), wt[ok][inside])
    return out
