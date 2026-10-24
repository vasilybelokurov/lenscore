"""Analytic limits of the lensing engine."""
import numpy as np

from lenscore import cosmology as C
from lenscore import lensing as Ln
from lenscore import population as Pop
from lenscore import profiles as P

ZL = 0.6
SCRIT, DL = C.sigma_crit(ZL)


def theta_e_sis(sig, zl=ZL):
    dcl, dcs = C.dist_mpc(zl), C.dist_mpc(C.ZS)
    return 4 * np.pi * (sig / C.C_KMS) ** 2 * (dcs - dcl) / dcs


def test_projection_point_mass_and_sis():
    pm = np.full((1, Ln.R3.size), 1e11)
    assert np.allclose(Ln.project(pm)[0], 1e11, rtol=1e-6)
    m2d = Ln.project(P.sis([200.0]))[0]
    # untruncated SIS: M_2D = (pi/2) M_3D; truncation at R3[-1] = 5 Mpc costs O(R / 5 Mpc)
    sel = (Ln.R2 > 1e-3) & (Ln.R2 < 10)
    assert np.allclose(m2d[sel], np.pi / 2 * 2 * 200.0**2 * Ln.R2[sel] / C.G, rtol=1e-3)


def test_sis_einstein_radius_caustic_and_separations():
    rows = P.sis([150.0, 250.0])
    te = Ln.einstein_radius(rows, DL, SCRIT)
    assert np.allclose(te, theta_e_sis(np.array([150.0, 250.0])), rtol=1e-3)
    beta_r, sep, _, _ = Ln.images(rows, DL, SCRIT)
    assert np.allclose(beta_r, te, rtol=2e-3)          # SIS cut at theta_E: cross-section pi theta_E^2
    assert np.allclose(sep, 2 * te[:, None] / C.ARCSEC, rtol=2e-3)


def test_sis_optical_depth_turner84():
    # constant comoving density n of SIS: tau = 16 pi^3 n (s/c)^4 D_c(z_s)^3 / 30 (flat universe)
    n, s = 1e-3, 200.0
    tau = Pop.optical_depth(P.sis([s]), [n]).sum()
    expect = 16 * np.pi**3 / 30 * n * (s / C.C_KMS) ** 4 * C.dist_mpc(C.ZS) ** 3
    assert abs(tau / expect - 1) < 2e-3


def test_flux_cut_keeps_sis_fraction():
    # SIS: q = (theta_E - beta) / (theta_E + beta), so q >= qmin keeps beta <= theta_E (1 - q)/(1 + q)
    qmin = 0.1
    full = Pop.optical_depth(P.sis([200.0]), [1e-3], nbeta=200).sum()
    cut = Pop.optical_depth(P.sis([200.0]), [1e-3], nbeta=200, qmin=qmin).sum()
    assert abs(cut / full / ((1 - qmin) / (1 + qmin)) ** 2 - 1) < 0.02


def test_sis_delays():
    _, sep, dphi, q = Ln.images(P.sis([200.0]), DL, SCRIT, nbeta=24, qmin=0.1)
    ok = np.isfinite(dphi[0])
    expect = 0.5 * (sep[0, ok] * C.ARCSEC) ** 2 * (1 - q[0, ok]) / (1 + q[0, ok])
    assert np.allclose(dphi[0, ok], expect, rtol=0.02)


def test_power_law_delays_scale_as_gamma_minus_one():
    # rho ∝ r^-gamma normalised to the SIS theta_E: dt = (D_dt / 2c)(theta_A^2 - theta_B^2)(gamma - 1)
    # near the Einstein radius (Kochanek 2002)
    re = theta_e_sis(200.0) * DL
    for gamma in (1.6, 2.0, 2.4):
        row = (Ln.R3 ** (3 - gamma))[None, :]
        row = row * np.pi * re**2 * SCRIT / np.interp(re, Ln.R2, Ln.project(row)[0])
        theta, _, _, alpha, g = Ln._profiles(row, DL, SCRIT)
        th, al = theta[0], alpha[0]
        psi = np.concatenate([[0], np.cumsum(0.5 * (al[1:] + al[:-1]) * np.diff(th))])
        b = 0.1 * re / DL
        ta, tb = Ln.outermost_root(g[0], th, [b])[0], Ln.outermost_root(g[0], th, [-b])[0]
        dphi = (0.5 * (tb + b) ** 2 - np.interp(tb, th, psi)) - (0.5 * (ta - b) ** 2 - np.interp(ta, th, psi))
        assert abs(dphi / (0.5 * (ta**2 - tb**2)) / (gamma - 1) - 1) < 0.015


def test_outermost_root_non_monotonic():
    th = np.linspace(0, 10, 1001)
    g = np.sin(th)                                      # roots of sin = 0.5 at ~0.52, 2.62, 6.81, 8.90
    assert abs(Ln.outermost_root(g, th, [0.5])[0] - (2 * np.pi + 5 * np.pi / 6)) < 1e-3
    assert np.isnan(Ln.outermost_root(g, th, [2.0])[0])


def test_per_row_histogram_sums_to_total():
    rows, w = P.sis([120.0, 200.0, 300.0]), [1e-3, 5e-4, 1e-4]
    tot = Pop.optical_depth(rows, w)
    per = Pop.optical_depth(rows, w, per_row=True)
    assert per.shape == (3,) + tot.shape and np.allclose(per.sum(axis=0), tot)
    h2 = Pop.optical_depth(rows, w, qmin=0.1, bins=Pop.SEP_E, dt_bins=Pop.DT_E)
    h2r = Pop.optical_depth(rows, w, qmin=0.1, bins=Pop.SEP_E, dt_bins=Pop.DT_E, per_row=True)
    assert np.allclose(h2r.sum(axis=0), h2)


def test_halo_response_dutton07():
    # nu = 0 leaves the halo unchanged, nu > 0 contracts and nu < 0 expands the inner dark matter,
    # mass is conserved at large r, and the solution satisfies r_f/r_i = [M_i(r_i)/M_f(r_f)]^nu
    from scipy.optimize import brentq
    m200, z, ms, re = 1e12, 0.5, 3e10, 3.0
    c = P.concentration(m200, z)
    mnfw, _ = P.nfw_m3d(m200, c, z)
    a, f = re / 1.8153, ms / m200
    r = np.array([0.5, 5.0, 50.0])
    m0, rmax = P.galaxy_halo_m3d(m200, c, z, ms, re, 0.0)
    assert np.allclose(m0(r) - ms * r**2 / (r + a) ** 2, (1 - f) * mnfw(r), rtol=1e-3)
    mc, _ = P.galaxy_halo_m3d(m200, c, z, ms, re, 1.0)
    me, _ = P.galaxy_halo_m3d(m200, c, z, ms, re, -0.5)
    assert np.all(mc(r[:2]) > m0(r[:2])) and np.all(me(r[:2]) < m0(r[:2]))
    assert np.allclose([mc(rmax), me(rmax)], m0(rmax), rtol=1e-2)
    for nu in (-0.5, 0.5, 1.0):
        m3, _ = P.galaxy_halo_m3d(m200, c, z, ms, re, nu)
        ri = np.array([0.01, 0.3, 3.0, 30.0])
        mdm = (1 - f) * mnfw(ri)
        rf = np.array([brentq(lambda x: m3(x) - ms * x**2 / (x + a) ** 2 - mm, 1e-6, 1e4) for mm in mdm])
        mf = ms * rf**2 / (rf + a) ** 2 + mdm
        assert np.allclose(rf / ri, (mnfw(ri) / mf) ** nu, rtol=2e-3)
