# %%
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from gnomproj import GnomonicProjection

# %%
# Set up the projection.
gp = GnomonicProjection(nside=256, size=256, taper=32, pad=32)

# %%
# Create a power spectrum.
logell = np.linspace(0, np.log(4096), 1000)
logcl = np.linspace(-7, -10, len(logell))
cl = 10**np.interp(np.log(np.arange(4097)), logell, logcl)
cl[0] = 0

# %%
# Generate a random map.
m = hp.synfast(cl, gp.nside, lmax=gp.nside*4)
hp.mollview(m)
plt.show()

# %%
# Project the map to patches.
patches = gp.healpix2patches(m)
plt.matshow(patches[0])
plt.show()

# %%
# Project the patches to the sky.
r = gp.patches2healpix(patches)
hp.mollview(r)
plt.show()

# %%
# Show the relative difference between the original map and the reconstructed map.
hp.mollview((r - m) / m.std())
plt.show()

# %%
# Show the power spectrum of the difference map.
plt.loglog(hp.anafast((r - m), lmax=gp.nside*2))
plt.show()

# %%
