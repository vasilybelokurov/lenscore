"""Enclosed-mass profiles on lensing.R3 [kpc -> Msun] and galaxy-halo recipes shared by
the populations."""
import numpy as np
from colossus.halo import concentration as _conc
from colossus.halo import mass_so

from .cosmology import G, H
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


MOSTER = dict(M10=11.590, M11=1.195, N10=0.0351, N11=-0.0247,
              b10=1.376, b11=-0.826, g10=0.608, g11=0.329)


def mstar_moster13(m200, z):
    """Stellar mass from Moster, Naab & White (2013, MNRAS 428, 3121), eqs. 2, 11-14."""
    p, a = MOSTER, z / (1 + z)
    m1 = 10 ** (p["M10"] + p["M11"] * a)
    n, b, g = p["N10"] + p["N11"] * a, p["b10"] + p["b11"] * a, p["g10"] + p["g11"] * a
    return 2 * n * m200 / ((m200 / m1) ** -b + (m200 / m1) ** g)
