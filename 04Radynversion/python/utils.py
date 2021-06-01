import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import PowerNorm,LinearSegmentedColormap
import matplotlib.ticker
import seaborn as sns

def intensity_vector(data, coord, coord_type):
    intensity_vec = np.zeros(data[0].header['NAXIS3'])

    if coord_type == "px":
        x = coord[0]
        y = coord[1]
    elif coord_type == "arcsec":
        x = int(round(coord[0]/data[0].header["CDELT1"]))
        y = int(round(coord[1]/data[0].header["CDELT2"]))

    for i in range(data[0].header["NAXIS3"]):
        intensity_vec[i] = data[0].data[i,y,x]

    return intensity_vec



def interp_to_radyn_grid(intensity_vector,centre_wvl,hw,wvl_range):
    '''
    A function to linearly interpolate the observational line profiles to the number of wavelength points in the RADYN grid.

    Parameters
    ----------
    intensity_vector : numpy.ndarray
        The intensity vector from a pixel in the CRISP image.
    centre_wvl : float
        The central measured wavelength obtained from the TWAVE1 keyword in the observartion's FITS header.
    hw : float
        The half-width of the line on the RADYN grid.
    wvl_range : numpy.ndarray
        The wavelength range from the observations.

    Returns
    -------
     : list
        A list of the interpolated wavelengths and intensities. Each element of the list is a numpy.ndarray.
    '''

    wvl_vector = np.linspace(centre_wvl-hw,centre_wvl+hw,num=30)
    interp = interp1d(wvl_range,intensity_vector,kind="linear")

    return [wvl_vector,interp(wvl_vector)]

def normalise(new_ca,new_ha):
    '''
    A function to normalise the spectral line profiles as the RADYN grid works on normalised profiles.

    Parameters
    ----------
    new_ca : numpy.ndarray
        The new calcium line interpolated onto the RADYN grid.
    new_ha : numpy.ndarray
        The new hydrogen line interpolated onto the RADYN grid.

    Returns
    -------
    new_ca : numpy.ndarray
        The interpolated calcium line normalised.
    new_ha : numpy.ndarray
        The interpolated hydrogen line normalised.
    '''

    peak_emission = max(np.amax(new_ca[1]),np.amax(new_ha[1]))

    new_ca[1] /= peak_emission
    new_ha[1] /= peak_emission

    return new_ca, new_ha

def inverse_velocity_conversion(out_velocities):
    '''
    A function to convert the calculated inverse velocities from the smooth space to the actual space.

    Parameters
    ----------
    out_velocities : torch.Tensor
        The velocity profiles obtained from the inversion.

    Returns
    -------
     : torch.Tensor
        The velocity profiles converted back to the actual space.
    '''

    v_sign = out_velocities / torch.abs(out_velocities)
    v_sign[torch.isnan(v_sign)] = 0

    return v_sign * (10**torch.abs(out_velocities) - 1.0)


def inversion_plots(results,z,ca_data,ha_data, rasterizeHists=True, figsize=None, powerNormIdx=0.3):
    '''
    A function to plot the results of the inversions.

    Parameters
    ----------
    results : dict
        The results from the inversions.m the latent space.
    z : torch.Tensor
        The height profiles of the RADYN grid.
    ca_data : list
        A concatenated list of the calcium wavelengths and intensities.
    ha_data : list
        A concatenated list of the hydrogen wavelengths and intensities.
    '''

    if figsize is None:
        figsize = (9,7)
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=figsize,constrained_layout=True)
    ax2 = ax[0].twinx()
    ca_wvls = ca_data[0]
    ha_wvls = ha_data[0]
    z_local = z


    z_edges = [z_local[0] - 0.5*(z_local[1]-z_local[0])]
    for i in range(z_local.shape[0]-1):
        z_edges.append(0.5*(z_local[i]+z_local[i+1]))
    z_edges.append(z_local[-1] + 0.5*(z_local[-1]-z_local[-2]))
    z_edges = [float(f) for f in z_edges]
    ca_edges = [ca_wvls[0] - 0.5*(ca_wvls[1]-ca_wvls[0])]
    for i in range(ca_wvls.shape[0]-1):
        ca_edges.append(0.5*(ca_wvls[i]+ca_wvls[i+1]))
    ca_edges.append(ca_wvls[-1] + 0.5*(ca_wvls[-1]-ca_wvls[-2]))
    ha_edges = [ha_wvls[0] - 0.5*(ha_wvls[1]-ha_wvls[0])]
    for i in range(ha_wvls.shape[0]-1):
        ha_edges.append(0.5*(ha_wvls[i]+ha_wvls[i+1]))
    ha_edges.append(ha_wvls[-1] + 0.5*(ha_wvls[-1]-ha_wvls[-2]))
    ne_edges = np.linspace(8,15,num=101)
    temp_edges = np.linspace(3,8,num=101)
    vel_max = 2*np.max(np.median(results["vel"],axis=0))
    vel_min = np.min(np.median(results["vel"],axis=0))
    vel_min = np.sign(vel_min)*np.abs(vel_min)*2
    vel_edges = np.linspace(vel_min,vel_max,num=101)

    # TODO(cmo): There is some not very tidy  code in here that could do with a good tidy.
    cmap_ne = [(1.0,1.0,1.0,0.1), (*sns.color_palette()[0], 1.0)]
    colors_ne = LinearSegmentedColormap.from_list('ne', cmap_ne)
    cmap_temp = [(1.0,1.0,1.0,0.1), (*sns.color_palette()[1], 1.0)]
    colors_temp = LinearSegmentedColormap.from_list('temp', cmap_temp)
    cmap_vel = [(1.0,1.0,1.0,0.1), (*sns.color_palette()[2], 1.0)]
    colors_vel = LinearSegmentedColormap.from_list('vel', cmap_vel)


    ax[0].hist2d(np.concatenate([z_local]*results["ne"].shape[0]),results["ne"].reshape((-1,)),bins=(z_edges,ne_edges),cmap=colors_ne,norm=PowerNorm(powerNormIdx), rasterized=rasterizeHists)
    ax[0].plot(z_local,np.median(results["ne"],axis=0), "--",c="k", zorder=3, linewidth=0.5)
    ax[0].set_ylabel(r"$\log{n_e}$ [\si{\centi\metre\tothe{-3}}]",color=cmap_ne[-1])
    ax[0].set_xlabel(r"$z$ [\si{\mega\metre}]")
    ax2.hist2d(np.concatenate([z_local]*results["temperature"].shape[0]),results["temperature"].reshape((-1,)),bins=(z_edges,temp_edges),cmap=colors_temp,norm=PowerNorm(powerNormIdx), rasterized=rasterizeHists)
    ax2.plot(z_local,np.median(results["temperature"],axis=0),"--",c="k", linewidth=0.5)
    ax2.set_ylabel(r"$\log{T}$ [\si{\kelvin}]",color=cmap_temp[-1])
    ax[1].hist2d(np.concatenate([z_local]*results["vel"].shape[0]),results["vel"].reshape((-1,)),bins=(z_edges,vel_edges),cmap=colors_vel,norm=PowerNorm(powerNormIdx), rasterized=rasterizeHists)
    ax[1].plot(z_local,np.median(results["vel"],axis=0),"--",c="k", linewidth=0.5)
    ax[0].set_xlim(None, 10.5)
    ax[1].set_xlim(None, 10.5)
    ax[1].set_ylabel(r"$v$ [\si{\kilo\metre\per\second}]",color=cmap_vel[-1])
    ax[1].set_xlabel(r"$z$ [\si{\mega\metre}]")

    return fig

z = np.array([-0.065, 0.016, 0.097, 0.178, 0.259, 0.340, 0.421, 0.502, 0.583, 0.664, 0.745, 0.826, 0.907, 0.988, 1.069, 1.150, 1.231, 1.312, 1.393, 1.474, 1.555, 1.636, 1.718, 1.799, 1.880, 1.961, 2.042, 2.123, 2.204, 2.285, 2.366, 2.447, 2.528, 2.609, 2.690, 2.771, 2.852, 2.933, 3.014, 3.095, 3.176, 3.257, 3.338, 3.419, 3.500, 4.360, 5.431, 6.766, 8.429, 10.5], dtype=np.float32)
