# The pinned fixtures: five real crystals, carried as RAW DATA so this file
# depends on no package type.  They are the magnetic rows of the integration
# roster (`test/integration/roster.jl`) minus the 2x2x2 supercell, which is the
# same content on 8x the cell and would multiply the pin's size for nothing.
#
#   bcc_Fe    Im-3m, one species, 3-body        -- the high-symmetry baseline
#   B2_FeRh   Pm-3m, two species, 3-body        -- two magnetic sublattices
#   hcp_Co    P6_3/mmc, non-orthogonal cell     -- two like sites, hexagonal
#   wz_GaN    P6_3mc, NO inversion centre       -- four atoms, no centre
#   rs_MnO    Fm-3m, a species with lmax = 0    -- the spin-free-species path
#
# Specs are pure spin (dense, sector-less): that is the surface both packages
# share, and the only one a pin can be written against once.

_pin_diag(a) = [a 0.0 0.0; 0.0 a 0.0; 0.0 0.0 a]

const PIN_FIXTURES = [
    (id = "bcc_Fe", L = _pin_diag(2.87), frac = [0.0 0.5; 0.0 0.5; 0.0 0.5],
     species = [1, 1], labels = ["Fe"], nbody = 3, lmax = [2], cutoff = 2.6),
    (id = "B2_FeRh", L = _pin_diag(2.99), frac = [0.0 0.5; 0.0 0.5; 0.0 0.5],
     species = [1, 2], labels = ["Fe", "Rh"], nbody = 3, lmax = [2, 2], cutoff = 2.7),
    (id = "hcp_Co",
     L = [2.507 -2.507/2 0.0; 0.0 2.507*sqrt(3)/2 0.0; 0.0 0.0 4.069],
     frac = [1/3 2/3; 2/3 1/3; 1/4 3/4], species = [1, 1], labels = ["Co"],
     nbody = 2, lmax = [2], cutoff = 2.6),
    (id = "wz_GaN",
     L = [3.189 -3.189/2 0.0; 0.0 3.189*sqrt(3)/2 0.0; 0.0 0.0 5.185],
     frac = [1/3 2/3 1/3 2/3; 2/3 1/3 2/3 1/3; 0.0 0.5 0.377 0.877],
     species = [1, 1, 2, 2], labels = ["Ga", "N"], nbody = 2, lmax = [1, 1],
     cutoff = 2.0),
    (id = "rs_MnO", L = _pin_diag(4.446),
     frac = [0.0 0.5 0.5 0.0 0.5 0.5 0.0 0.0;
             0.0 0.5 0.0 0.5 0.5 0.0 0.5 0.0;
             0.0 0.0 0.5 0.5 0.5 0.0 0.0 0.5],
     species = [1, 1, 1, 1, 2, 2, 2, 2], labels = ["Mn", "O"], nbody = 2,
     lmax = [2, 0], cutoff = 3.2),
]
