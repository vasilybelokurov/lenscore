"""Cosmology and units used throughout.

Flat LCDM, H0 = 70 km/s/Mpc, Omega_m = 0.3 (Omega_b = 0.048, sigma_8 = 0.81, n_s = 0.96 for
the colossus halo functions). Units: kpc, Msun, km/s; angles in radians unless named *_arcsec.
Importing this module makes this cosmology colossus's current one.
"""
import numpy as np
from colossus.cosmology import cosmology

COSMO = cosmology.setCosmology("quells", dict(flat=True, H0=70.0, Om0=0.3, Ob0=0.048,
                                              sigma8=0.81, ns=0.96))
H = 0.7
FB = 0.048 / 0.3                    # cosmic baryon fraction
G = 4.30091e-6                      # kpc (km/s)^2 / Msun
C_KMS = 299792.458
ARCSEC = np.pi / 180 / 3600
ZS = 2.0                            # default source redshift
MPC_M = 3.0857e22
DAY = 86400.0


def dist_mpc(z):
    """Comoving distance [Mpc]."""
    return COSMO.comovingDistance(0.0, z) / H


def sigma_crit(zl, zs=ZS):
    """Critical surface density [Msun / kpc^2] and the lens angular-diameter distance [kpc]."""
    dcl, dcs = dist_mpc(zl), dist_mpc(zs)
    dl, ds, dls = dcl / (1 + zl), dcs / (1 + zs), (dcs - dcl) / (1 + zs)
    return C_KMS**2 / (4 * np.pi * G) * ds / (dl * dls) / 1e3, dl * 1e3


def dt_distance_m(zl, zs=ZS):
    """Time-delay distance (1 + z_l) D_l D_s / D_ls [m]."""
    dcl, dcs = dist_mpc(zl), dist_mpc(zs)
    return (1 + zl) * (dcl / (1 + zl)) * (dcs / (1 + zs)) / ((dcs - dcl) / (1 + zs)) * MPC_M


def dvdz(z):
    """Comoving volume per unit redshift per steradian [Mpc^3]."""
    return C_KMS / COSMO.Hz(z) * dist_mpc(z) ** 2


def rho_crit(z):
    """Critical density [Msun / kpc^3]."""
    return COSMO.rho_c(z) * H**2
