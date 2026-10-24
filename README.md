# lenscore

Spherical strong-lensing engine: multiple-imaging cross-sections, image separations, time delays and
the optical depth of lens populations. Used by
[quells-lensdb](https://github.com/vasilybelokurov/quells-lensdb) (sub-arcsecond lensed quasars and their
time delays, supporting the QUELLS paper).

A lens is its enclosed 3D mass M(<r) on the radial grid `lensing.R3` (kpc, Msun).

| module | contents |
|---|---|
| `cosmology` | flat LCDM, H0 = 70, Omega_m = 0.3 (colossus); distances, Sigma_crit, time-delay distance, dV/dz |
| `lensing` | projection, Einstein radius, cross-section pi beta_r^2, outer-image separations, Fermat delays, flux ratios |
| `population` | optical depth over lens redshift, histogrammed in separation or (separation, delay) |
| `profiles` | SIS, inner power law, NFW, Hernquist, Moster+13 SHMR, Diemer+19 c(M) |

Importing `lenscore.cosmology` sets colossus's current cosmology.

## Install and test

    pip install git+https://github.com/vasilybelokurov/lenscore
    # or, for development:
    git clone https://github.com/vasilybelokurov/lenscore && cd lenscore
    pip install -e ".[test]" && python -m pytest -q

The tests check analytic limits: SIS projection, Einstein radius, caustic and separations,
the Turner et al. (1984) SIS optical depth, the flux-cut area fraction ((1-q)/(1+q))^2, SIS
delays, and the delay scaling (gamma - 1) for power laws (Kochanek 2002).
