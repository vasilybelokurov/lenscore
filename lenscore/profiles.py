"""Enclosed-mass profiles on lensing.R3 [kpc -> Msun] and galaxy-halo recipes shared by
the populations."""
import numpy as np
from colossus.halo import concentration as _conc
from colossus.halo import mass_so

from .cosmology import G, H, rho_crit
from .lensing import R3


def sis(sig):
    """Singular isothermal spheres, M(<r) = 2 sigma^2 r / G, one row per sigma [km/s]."""
    return 2 * np.asarray(sig, float)[:, None] ** 2 * R3[None, :] / G


def power_law(sig, gamma, re):
    """rho ∝ r^-gamma inside re, isothermal outside, with M(<re) equal to the SIS value
    2 sigma^2 re / G; gamma = 2 is the SIS."""
    m_re = 2 * np.asarray(sig, float) ** 2 * re / G
    r = R3[None, :]
    return np.where(r < re, m_re[:, None] * (r / re) ** (3 - gamma), m_re[:, None] * (r / re))


def nfw_m(r, rs, c):
    """NFW enclosed mass in units of M200 (scale radius rs, concentration c)."""
    x = r / rs
    mu = lambda y: np.log1p(y) - y / (1 + y)
    return mu(x) / mu(c)


def hernquist_m(r, a):
    """Hernquist enclosed mass in units of the total (scale radius a = R_e / 1.8153)."""
    return r**2 / (r + a) ** 2


def concentration(m200, z):
    """c200c(M) of Diemer & Joyce (2019) via colossus; m200 in Msun."""
    return _conc.concentration(np.asarray(m200) * H, "200c", z, model="diemer19")


def r200(m200, z):
    """r_200c [kpc] for m200 in Msun."""
    return mass_so.M_to_R(np.asarray(m200) * H, z, "200c") / H


def nfw_m3d(m200, c, z):
    """Enclosed-mass function of an NFW halo with mass m200 (200 x critical) and concentration c,
    and its r200 [kpc]."""
    r200 = (3 * m200 / (4 * np.pi * 200 * rho_crit(z))) ** (1 / 3)
    rs = r200 / c
    return (lambda r: m200 * nfw_m(r, rs, c)), r200


def galaxy_halo_m3d(m200, c, z, mstar, re, nu, rmax_fac=10.0):
    """Total M(<r) of a Hernquist galaxy (a = re / 1.8153) inside an NFW halo with halo response.

    Response of Dutton et al. (2007, ApJ 654, 27): r_f/r_i = [M_i(r_i)/M_f(r_f)]^nu, with nu = 1
    the adiabatic contraction of Blumenthal et al. (1986), nu = 0 no response, nu < 0 expansion.
    The condensed baryons are the stars (a fraction f = mstar/m200 of the initial halo); the dark
    matter is (1 - f) of the NFW halo, extended to rmax_fac x r200.
    Returns (callable M(<r), rmax).
    """
    m_nfw, r200 = nfw_m3d(m200, c, z)
    a = re / 1.8153
    m_star = lambda r: mstar * r**2 / (r + a) ** 2
    f = mstar / m200
    ri = np.logspace(-4, np.log10(rmax_fac * r200), 600)
    mi = m_nfw(ri)
    # vectorised bisection in ln r_f of ln(r_f/r_i) - nu ln[M_i/M_f(r_f)] = 0 (monotonic in r_f)
    lo, hi = np.log(ri) - 28.0, np.log(ri) + 28.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        up = mid - np.log(ri) - nu * np.log(mi / (m_star(np.exp(mid)) + (1 - f) * mi)) > 0
        hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
    lr, lm = 0.5 * (lo + hi), np.log((1 - f) * mi)
    m_dm = lambda r: np.exp(np.interp(np.log(r), lr, lm, left=-np.inf))
    rmax = np.exp(lr[-1])
    return (lambda r: m_star(np.minimum(r, rmax)) + m_dm(np.minimum(r, rmax))), rmax


def re_shen03(mstar):
    """Early-type half-light radius [kpc], Shen et al. (2003, MNRAS 343, 978): 2.88e-6 M^0.56."""
    return 2.88e-6 * mstar**0.56


MOSTER = dict(M10=11.590, M11=1.195, N10=0.0351, N11=-0.0247,
              b10=1.376, b11=-0.826, g10=0.608, g11=0.329)


def mstar_moster13(m200, z):
    """Stellar mass from Moster, Naab & White (2013, MNRAS 428, 3121), eqs. 2, 11-14."""
    p, a = MOSTER, z / (1 + z)
    m1 = 10 ** (p["M10"] + p["M11"] * a)
    n, b, g = p["N10"] + p["N11"] * a, p["b10"] + p["b11"] * a, p["g10"] + p["g11"] * a
    return 2 * n * m200 / ((m200 / m1) ** -b + (m200 / m1) ** g)
