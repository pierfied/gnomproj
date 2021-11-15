import healpy as hp
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

class GnomonicProjection:
    '''
    Class for performing gnomonic projection.
    '''

    def __init__(self, nside, size, taper=0, pad=0, resol=None):
        '''
        Initialize the class.

        Parameters
        ----------
        nside : int
            The nside of the healpix map.
        size : int
            The size of the output patches.
        taper : int
            The size of the linear tapering window.
        pad : int
            The size of the padding window where weights are zero.
        resol : float
            The resolution of the output patches. Defaults to half the resolution of the input map.
        '''

        self.nside = nside
        self.size = size
        self.taper = taper
        self.pad = pad
        if resol is None:
            self.resol = hp.nside2resol(nside) / 2
        else:
            self.resol = resol

        self.compute_weights()
        
    def sph2gnom(self, theta, phi, theta0, phi0):
        '''
        Convert from spherical to gnomonic coordinates.

        Parameters
        ----------
        theta : float
            The theta coordinate.
        phi : float
            The phi coordinate.
        theta0 : float
            The theta coordinate of the center of the gnomonic projection.
        phi0 : float
            The phi coordinate of the center of the gnomonic projection.

        Returns
        -------
        x : float
            The x coordinate in the gnomonic projection.
        y : float
            The y coordinate in the gnomonic projection.
        '''

        theta0 = theta0 - np.pi / 2
        theta = theta - np.pi / 2

        cosc = np.sin(theta0) * np.sin(theta) + np.cos(theta0) * np.cos(theta) * np.cos(phi - phi0)

        x = np.cos(theta) * np.sin(phi - phi0) / cosc
        y = (np.cos(theta0) * np.sin(theta) - np.sin(theta0) * np.cos(theta) * np.cos(phi - phi0)) / cosc

        return x, y

    def gnom2sph(self, x, y, theta0, phi0):
        '''
        Convert from gnomonic to spherical coordinates.

        Parameters
        ----------
        x : float
            The x coordinate in the gnomonic projection.
        y : float
            The y coordinate in the gnomonic projection.
        theta0 : float
            The theta coordinate of the center of the gnomonic projection.
        phi0 : float
            The phi coordinate of the center of the gnomonic projection.

        Returns
        -------
        theta : float
            The theta coordinate.
        phi : float
            The phi coordinate.
        '''

        theta0 = theta0 - np.pi / 2

        rho = np.sqrt(x**2 + y**2)
        c = np.arctan(rho)

        theta = np.arcsin(np.cos(c) * np.sin(theta0) + y * np.sin(c) * np.cos(theta0) / (rho + 1e-8)) + np.pi / 2
        phi = phi0 + np.arctan2(x * np.sin(c), rho * np.cos(theta0) * np.cos(c) - y * np.sin(theta0) * np.sin(c))

        return theta, phi

    def sph2gnom_patch_weights(self, patch_num, theta0, phi0):
        '''
        Compute the weights for the gnomonic projection for a single patch.

        Parameters
        ----------
        patch_num : int
            The patch number.
        theta0 : float
            The theta coordinate of the center of the gnomonic projection.
        phi0 : float
            The phi coordinate of the center of the gnomonic projection.

        Returns
        -------
        rows : int
            The row indices of the sparse matrix.
        cols : int
            The column indices of the sparse matrix.
        weights : float
            The weights of the sparse matrix.
        '''

        # Compute the spherical coordinates of the patch pixels.
        theta, phi = self.gnom2sph(self.x, self.y, theta0, phi0)

        # Compute the pixel indices of the patch pixels and the weights.
        rows = np.arange(self.size**2) + patch_num * self.size**2
        cols = hp.ang2pix(self.nside, theta, phi).ravel()
        weights = np.ones(self.size**2)

        return rows, cols, weights

    def gnom2sph_patch_weights(self, patch_num, theta0, phi0):
        '''
        Compute the weights for the spherical projection for a single patch.

        Parameters
        ----------
        patch_num : int
            The patch number.
        theta0 : float
            The theta coordinate of the center of the gnomonic projection.
        phi0 : float
            The phi coordinate of the center of the gnomonic projection.

        Returns
        -------
        rows : int
            The row indices of the sparse matrix.
        cols : int
            The column indices of the sparse matrix.
        weights : float
            The weights of the sparse matrix.
        '''

        # Compute the spherical coordinates of the patch pixels.
        theta, phi = self.gnom2sph(self.x, self.y, theta0, phi0)
        pix = hp.ang2pix(self.nside, theta, phi)
        rows = pix.ravel()
        cols = np.arange(self.size**2) + patch_num * self.size**2

        # Compute the weights, including the tapering and padding.
        weights = np.ones([self.size, self.size])
        pad_weights = np.zeros(self.pad)
        taper_weights = np.linspace(0, 1, self.taper + 2)[1:-1]
        pad_taper_weights = np.concatenate([pad_weights, taper_weights])
        for i in range(self.pad + self.taper):
            weights[i, :] = pad_taper_weights[i]
            weights[:, i] = pad_taper_weights[i]
            weights[-i - 1, :] = pad_taper_weights[i]
            weights[:, -i - 1] = pad_taper_weights[i]

        return rows, cols, weights.ravel()
    
    def compute_weights(self):
        '''
        Compute the weights for the spherical and gnomonic projections.
        '''

        # Calculate the spacing and x, y coordinates of the patch pixels for the given resolution.
        self.spacing = (np.arange(self.size) - self.size / 2 + 0.5) * self.resol
        self.x, self.y = np.meshgrid(self.spacing, self.spacing, indexing='xy')

        # Compute the rings in the theta direction required to cover the sky.
        delta_theta = 2 * self.gnom2sph((self.size//2 - self.taper - self.pad) * self.resol, 0, 0, 0)[0]
        nrings = int(np.ceil(np.pi / delta_theta)) + 1
        ring_thetas = np.linspace(0, np.pi, nrings)

        # Compute the values of phi required for each ring to cover the sky.
        theta0 = []
        phi0 = []
        for i, theta in enumerate(ring_thetas):
            # Check if at the poles, and if so, only use one patch.
            if i == 0 or i == nrings - 1:
                npatches = 1
            else:
                # Find the minimum length along phi for the ring and ensure the patches are separated by less than that.
                _, phi = self.gnom2sph(self.x, self.y, theta, 0)
                delta_phi = phi[:,-1 - self.taper - self.pad].min() - phi[:,self.taper + self.pad].max()
                npatches = int(np.ceil(2 * np.pi / delta_phi))
            
            ring_phis = np.linspace(0, 2 * np.pi, npatches + 1)[:-1]

            theta0.append(theta * np.ones(npatches))
            phi0.append(ring_phis)
        
        theta0 = np.concatenate(theta0)
        phi0 = np.concatenate(phi0)

        npatches = len(theta0)

        # Compute the weights for the spherical to gnomonic projection.
        rows, cols, weights = zip(*Parallel(n_jobs=-1)(delayed(self.sph2gnom_patch_weights)(i, theta0[i], phi0[i]) for i in tqdm(range(npatches), desc='Computing Healpix -> Patch Weights')))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        self.sph2gnom_weights = csr_matrix((weights, (rows, cols)), shape=(npatches * self.size**2, hp.nside2npix(self.nside)))

        # Compute the weights for the gnomonic to spherical projection.
        rows, cols, weights = zip(*Parallel(n_jobs=-1)(delayed(self.gnom2sph_patch_weights)(i, theta0[i], phi0[i]) for i in tqdm(range(npatches), desc='Computing Patch -> Healpix Weights')))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)

        # Ensure that the weights for each pixel are normalized.
        inds = np.argsort(rows)
        rows = rows[inds]
        cols = cols[inds]
        weights = weights[inds]
        right = np.concatenate([np.where(np.diff(rows))[0] + 1, [hp.nside2npix(self.nside)]])
        left = np.concatenate([[0], right[:-1]])
        for i, (l, r) in enumerate(tqdm(zip(left, right), total=len(left), desc='Normalizing Patch -> Healpix Weights')):
            weights[l:r] /= np.sum(weights[l:r])
        self.gnom2sph_weights = csr_matrix((weights, (rows, cols)), shape=(hp.nside2npix(self.nside), npatches * self.size**2))
    
    def healpix2patches(self, m):
        '''
        Convert the healpix map to gnomonic patches.

        Parameters
        ----------
        m : float
            The healpix map.

        Returns
        -------
        patches : float
            The gnomonic patches.
        '''

        return (self.sph2gnom_weights @ m).reshape(-1, self.size, self.size)
    
    def patches2healpix(self, patches):
        '''
        Convert the gnomonic patches to healpix map.

        Parameters
        ----------
        patches : float
            The gnomonic patches.

        Returns
        -------
        m : float
            The healpix map.
        '''

        return self.gnom2sph_weights @ patches.ravel()
