import argparse
import json
import logging
import os

import galsim
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift
from tqdm import tqdm
import csv

from utils.utils_data import down_sample


def get_LSST_PSF(atmos_fwhm, atmos_e, atmos_beta,
                 g1_err=0, g2_err=0,
                 fov_pixels=48, pixel_scale=0.2, upsample=4):
    """Simulate a PSF from a ground-based observation (typically LSST). The PSF consists of an optical component and an atmospheric component.

    Args:
        lam_over_diam (float): Wavelength over diameter of the telescope.
        opt_defocus (float): Defocus in units of incident light wavelength.
        opt_c1 (float): Coma along y in units of incident light wavelength.
        opt_c2 (float): Coma along x in units of incident light wavelength.
        opt_a1 (float): Astigmatism (like e2) in units of incident light wavelength. 
        opt_a2 (float): Astigmatism (like e1) in units of incident light wavelength. 
        opt_obscuration (float): Linear dimension of central obscuration as fraction of pupil linear dimension, [0., 1.).
        atmos_fwhm (float): The full width at half maximum of the Kolmogorov function for atmospheric PSF.
        atmos_e (float): Ellipticity of the shear to apply to the atmospheric component.
        atmos_beta (float): Position angle (in radians) of the shear to apply to the atmospheric component, twice the phase of a complex valued shear.
        spher (float): Spherical aberration in units of incident light wavelength.
        trefoil1 (float): Trefoil along y axis in units of incident light wavelength.
        trefoil2 (float): Trefoil along x axis in units of incident light wavelength.
        g1_err (float, optional): The first component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        g2_err (float, optional): The second component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for the PSF image. Defaults to `4`.

    Returns:
        `torch.Tensor`: Simulated PSF image with shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """

    # Atmospheric PSF
    atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1)
    atmos = atmos.shear(e=atmos_e, beta=atmos_beta*galsim.radians)

    #remove the optics component
    psf = atmos
    
    # Shear the overall PSF to simulate a erroneously estimated PSF when necessary.
    psf = psf.shear(g1=g1_err, g2=g2_err) 

    # Draw PSF images.
    psf_image = galsim.ImageF(fov_pixels*upsample, fov_pixels*upsample)
    psf.drawImage(psf_image, scale=pixel_scale/upsample, method='auto')
    psf_image = torch.from_numpy(psf_image.array)
         
    return psf_image


def get_COSMOS_Galaxy(real_galaxy_catalog, idx, 
                      gal_g, gal_beta, theta, gal_mu, dx, dy, 
                      fov_pixels=48, pixel_scale=0.2, upsample=4):
    """Simulate a background galaxy with data from COSMOS Catalog.

    Args:
        real_galaxy_catalog (`galsim.COSMOSCatalog`): A `galsim.RealGalaxyCatalog` object, from which the parametric galaxies are read out.
        idx (int): Index of the chosen galaxy in the catalog.
        gal_flux (float): Total flux of the galaxy in the simulated image.
        sky_level (float): Skylevel in the simulated image.
        gal_g (float): The shear to apply.
        gal_beta (float): Position angle (in radians) of the shear to apply, twice the phase of a complex valued shear.
        theta (float): Rotation angle of the galaxy (in radians, positive means anticlockwise).
        gal_mu (float): The lensing magnification to apply.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for galaxy image. Defaults to `4`.

    Returns:
        `torch.Tensor`: Simulated galaxy image of shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """

    # Read out real galaxy from the catalog.
    gal_ori = galsim.RealGalaxy(real_galaxy_catalog, index = idx)
    
    # Add random rotation, shear, and magnification.
    gal = gal_ori.rotate(theta * galsim.radians) # Rotate by a random angle
    gal = gal.shear(g=gal_g, beta=gal_beta * galsim.radians) # Apply the desired shear
    gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.
    
    # Draw galaxy image.
    gal_image = galsim.ImageF(fov_pixels*upsample, fov_pixels*upsample)
    psf_hst = real_galaxy_catalog.getPSF(idx)
    gal = galsim.Convolve([psf_hst, gal]) # Concolve wth original PSF of HST.
    gal.drawImage(gal_image, scale=pixel_scale/upsample, offset=(dx,dy), method='auto')
        
    gal_image = torch.from_numpy(gal_image.array) # Convert to PyTorch.Tensor.
    gal_image = torch.max(gal_image, torch.zeros_like(gal_image))
    
    return gal_image


def generate_data_deconv(data_path, n_train=40000, load_info=True,
                         survey='LSST', I='23.5', fov_pixels=48, pixel_scale=0.2, upsample=4,
                         snrs=[20, 40, 60, 80, 100, 150, 200],
                         shear_errs=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
                         fwhm_errs=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]):
    """Generate simulated galaxy images and corresponding PSFs for deconvolution.

    Args:
        data_path (str): Path to save the dataset. 
        train_split (float, optional): Proportion of data used in train dataset, the rest will be used in test dataset. Defaults to `0.7`.
        survey (str, optional): _description_. Defaults to `'LSST'`.
        I (str, optional): The sample in COSMOS data to use, `"23.5"` or `"25.2"`. Defaults to `"23.5"`.
        fov_pixels (int, optional):  Size of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale in arcsec of the images. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for simulations. Defaults to `4`.
        snrs (list, optional): The list of SNR to be simulated for testing. Defaults to `[10, 15, 20, 40, 60, 80, 100, 150, 200]`.
        shear_errs (list, optional): The list of systematic PSF shear error to be simulated for testing. Defaults to `[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]`.
        fwhm_errs (list, optional): The list of systematic PSF FWHM error to be simulated for testing. Defaults to `[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]`.
    """
    
    logger = logging.getLogger('DataGenerator')
    logger.info('Simulating %s images for deconvolution using I=%s COSMOS data.', survey, I)
    
    # Create directory for the dataset.
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(os.path.join(data_path, 'obs')):
        os.mkdir(os.path.join(data_path, 'obs'))
    if not os.path.exists(os.path.join(data_path, 'gt')):
        os.mkdir(os.path.join(data_path, 'gt'))
    if not os.path.exists(os.path.join(data_path, 'psf')):
        os.mkdir(os.path.join(data_path, 'psf'))
    if not os.path.exists(os.path.join(data_path, 'visualization')): 
        os.mkdir(os.path.join(data_path, 'visualization'))


    n_total = 56030
    info_file = os.path.join(data_path, f'info.json')

    if load_info:
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            survey = info['survey']
            sequence = info['sequence']
            I = info['I']
            pixel_scale = info['pixel_scale']
            n_total, n_train, n_test = info['n_total'], info['n_train'], info['n_test']
            logger.info(' Successfully loaded dataset information from %s.', info_file)
        except:
            raise Exception(' Failed loading dataset information from %s.', info_file)
    else:
        sequence = np.arange(0, n_total) # Generate random sequence for dataset.
        np.random.shuffle(sequence)
        info = {'survey':survey, 'I':I, 'fov_pixels':fov_pixels, 'pixel_scale':pixel_scale,
                'n_total':n_total, 'n_train':n_train, 'n_test':n_total - n_train, 'sequence':sequence.tolist()}
        with open(info_file, 'w') as f:
            json.dump(info, f)
        logger.info(' Dataset information saved to %s.', info_file)

    # Random number generators for the parameters.
    random_seed = 31415
    rng_base = galsim.BaseDeviate(seed=random_seed)
    rng = galsim.UniformDeviate(seed=random_seed) # U(0,1).
    fwhms = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    freqs = np.array([0., 20., 17., 13., 9., 0.])
    fwhm_table = galsim.LookupTable(x=fwhms, f=freqs, interpolant='spline')
    fwhms = np.linspace(fwhms[0], fwhms[-1], 100) # Upsample the distribution.
    freqs = np.array([fwhm_table(fwhm) for fwhm in fwhms]) / fwhm_table.integrate() # Normalization.
    rng_fwhm = galsim.DistDeviate(seed=rng_base, function=galsim.LookupTable(x=fwhms, f=freqs, interpolant='spline'))
    
    
    # CCD and sky parameters.
    exp_time = 30.                          # Exposure time (2*15 seconds).
    sky_brightness = 20.48                  # Sky brightness (absolute magnitude) in i band.
    zero_point = 27.85                      # Instrumental zero point, i.e. asolute magnitude that would produce one e- per second.
    gain = 2.3                              # CCD Gain (e-/ADU).
    qe = 0.94                               # CCD Quantum efficiency.
    
    
    for k in tqdm(range(0, n_total)):

        # Atmospheric PSF
        atmos_fwhm = rng_fwhm()             # Atmospheric seeing (arcsec), the FWHM of the Kolmogorov function.
        atmos_e = 0.01 + 0.02 * rng()       # Ellipticity of atmospheric PSF (magnitude of the shear in the “distortion” definition), U(0.01, 0.03).
        atmos_beta = 2. * np.pi * rng()     # Shear position angle (radians), N(0,2*pi).
        

        data = [k, atmos_fwhm, atmos_e, atmos_beta] # id, atmos_fwhm, atoms_e, atmos_beta
        with open('/home/yuchenwang/outputs/parameters.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        psf_image = get_LSST_PSF(atmos_fwhm, atmos_e, atmos_beta, 
                                 0, 0,
                                 fov_pixels, pixel_scale, upsample) 

        psf = down_sample(psf_image.clone(), upsample)
  
        torch.save(psf.clone(), os.path.join(data_path, 'psf', f"psf_{k}.pth"))
       
          

            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for dataset.')
    parser.add_argument('--task', type=str, default='Deconv', choices=['Deconv', 'Denoise'])
    parser.add_argument('--n_train', type=int, default=40000)
    parser.add_argument('--load_info', action="store_true")
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=str, default='23.5', choices=['23.5', '25.2'])
    parser.add_argument('--fov_pixels', type=int, default=48)
    parser.add_argument('--pixel_scale', type=float, default=0.2)
    parser.add_argument('--upsample', type=int, default=4)
    opt = parser.parse_args()
    
    if opt.task == 'Deconv':
        generate_data_deconv(data_path='/mnt/WD6TB/yuchen_wang/dataset/LSST_23.5_EM/', n_train=opt.n_train, load_info=opt.load_info,
                             survey=opt.survey, I=opt.I, fov_pixels=opt.fov_pixels, pixel_scale=opt.pixel_scale, upsample=opt.upsample,
                             snrs=[20, 40, 60, 80, 100, 150, 200],
                             shear_errs=[0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
                             fwhm_errs=[0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2])
    else:
        raise ValueError('Invalid task.')
    