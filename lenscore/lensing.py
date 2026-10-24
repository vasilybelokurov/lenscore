"""Spherical lenses specified by their enclosed 3D mass M(<r) on the radial grid R3.

Projection: M_2D(<R) = sum over shells of dM(r) f(R, r), with f = 1 for r <= R and
f = 1 - sqrt(1 - R^2/r^2) for r > R, on the projected radii R2.
Lens equation along a line through the centre: beta = g(theta) = theta - alpha(theta),
alpha = theta kbar. A source is multiply imaged if beta < beta_r = max(alpha - theta): the
radial caustic for cored or cuspy profiles, the cut for an SIS. Cross-section pi beta_r^2.
Images of each source sample are the outermost roots of g = +beta (A) and g = -beta (B),
found by bracketing, since g need not be monotonic beyond its minimum.
Delay: phi_B - phi_A with phi = (theta - beta)^2 / 2 - psi, psi = int alpha dtheta.
Magnification: 1 / mu = (1 - kbar)(1 - 2 kappa + kbar).
"""
import numpy as np

from .cosmology import ARCSEC

R3 = np.logspace(-3.5, 3.7, 900)                 # kpc, 3D radii (cell edges)
RMID = np.sqrt(R3[1:] * R3[:-1])
R2 = np.logspace(-3.3, 2.3, 500)                 # kpc, projected radii
_x = np.clip(R2[:, None] / RMID[None, :], 0, 1)
PROJ = np.where(R2[:, None] >= RMID[None, :], 1.0, 1 - np.sqrt(1 - _x**2))


def project(rows):
    """Projected mass M_2D(<R2) for enclosed-mass rows on R3, shape (N, len(R2))."""
    dm = np.diff(rows, axis=1)
    dm[:, 0] += rows[:, 0]
    return dm @ PROJ.T


def outermost_root(g, theta, values):
    """theta of the outermost solution of g(theta) = v for each v (linear in a bracket); NaN if none."""
    out = np.full(len(values), np.nan)
    for k, v in enumerate(values):
        d = g - v
        idx = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if idx.size:
            a = idx[-1]
            out[k] = theta[a] + (theta[a + 1] - theta[a]) * d[a] / (d[a] - d[a + 1])
    return out


def _profiles(rows, dl_kpc, scrit):
    n = len(rows)
    dl = np.broadcast_to(np.atleast_1d(dl_kpc), (n,))
    sc = np.broadcast_to(np.atleast_1d(scrit), (n,))
    m2d = project(rows)
    kbar = m2d / (np.pi * R2**2 * sc[:, None])
    kap = 0.5 * kbar * np.gradient(np.log(np.maximum(m2d, 1e-30)), np.log(R2), axis=1)
    theta = R2[None, :] / dl[:, None]                                    # rad
    alpha = theta * kbar
    return theta, kbar, kap, alpha, theta - alpha


def einstein_radius(rows, dl_kpc, scrit):
    """Outermost radius where the mean convergence is 1 [rad]; NaN if none."""
    theta, _, _, _, g = _profiles(rows, dl_kpc, scrit)
    return np.array([outermost_root(g[i], theta[i], [0.0])[0] for i in range(len(rows))])


def images(rows, dl_kpc, scrit, nbeta=8, qmin=None):
    """The two outer images of nbeta equal-area source positions per lens.

    Returns beta_r [rad], shape (N,), and (N, nbeta) arrays of the image separation
    [arcsec], the Fermat-potential difference phi_B - phi_A [rad^2] and the flux ratio
    q = |mu_faint / mu_bright|, NaN where a sample does not count. Every counted sample
    carries cross-section pi beta_r^2 / nbeta.
    qmin: count only samples with q >= qmin, and cap the sampled region at 2 theta_E so
    that divergent central deflections (gamma > 2) do not push every sample to large beta
    (no effect for an SIS).
    """
    theta, kbar, kap, alpha, g = _profiles(rows, dl_kpc, scrit)
    psi = np.concatenate([np.zeros((len(rows), 1)),
                          np.cumsum(0.5 * (alpha[:, 1:] + alpha[:, :-1]) * np.diff(theta, axis=1), axis=1)],
                         axis=1)
    inv_mu = (1 - kbar) * (1 - 2 * kap + kbar)
    pb = (np.arange(nbeta) + 0.5) / nbeta
    beta_r = np.zeros(len(rows))
    sep, dphi, q = (np.full((len(rows), nbeta), np.nan) for _ in range(3))
    for i in range(len(rows)):
        br = -g[i].min()
        if br <= 0:
            continue
        if qmin is not None:
            te = outermost_root(g[i], theta[i], [0.0])[0]
            if np.isfinite(te):
                br = min(br, 2 * te)
        beta = br * np.sqrt(pb)
        ta = outermost_root(g[i], theta[i], beta)
        tb = outermost_root(g[i], theta[i], -beta)
        mua = 1 / np.abs(np.interp(ta, theta[i], inv_mu[i]))
        mub = 1 / np.abs(np.interp(tb, theta[i], inv_mu[i]))
        qi = np.minimum(mua, mub) / np.maximum(mua, mub)
        ok = np.isfinite(ta) & np.isfinite(tb)
        if qmin is not None:
            ok &= qi >= qmin
        phia = 0.5 * (ta - beta) ** 2 - np.interp(ta, theta[i], psi[i])
        phib = 0.5 * (tb + beta) ** 2 - np.interp(tb, theta[i], psi[i])
        beta_r[i] = br
        sep[i, ok] = (ta[ok] + tb[ok]) / ARCSEC
        dphi[i, ok] = (phib - phia)[ok]
        q[i, ok] = qi[ok]
    return beta_r, sep, dphi, q
