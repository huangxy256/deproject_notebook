import numpy as np
from copy import deepcopy

from deproject.Profiles.SIS_truncated_physical import SIS_truncated_physical
from deproject.Profiles.Hernquist import Hernquist
from deproject.Cosmo.default_cosmo import get_default_lens_cosmo
from jampy.jam_axi_proj import jam_axi_proj
from jampy.mge_half_light_isophote import mge_half_light_isophote
from deproject.MGE_analysis.mge_proj import MGE_Proj
from deproject.MGE_analysis.intr_mge import Intr_MGE

from scipy.stats import truncnorm

def qintr_prior(mu, sigma, num): # sample q_intr from truncated Gaussian prior
    lo_lim, hi_lim = 0, 1
    lo_transformed = (lo_lim - mu) / sigma
    hi_transformed = (hi_lim - mu) / sigma
    qintr_prior = truncnorm(lo_transformed, hi_transformed, loc = mu, scale = sigma)
    return qintr_prior.rvs(size = num)

def isotropic_inc(num):
    cos_i = np.random.uniform(0, 1, num)
    return np.arccos(cos_i)

def qobs_model(qintr, inc): # projeciton formula
    qobs = np.sqrt((qintr * np.sin(inc))**2 + np.cos(inc)**2)
    return qobs

def get_truncsis_intr_mge(sigma_v, rc_kpc, r_sym, qintr, plot_mge=0, fignum=1):
    """get the amplitude and dispersion of the MGE describing the INTRINSIC mass density/stellar light profile along the symmetry axis 

    Args:
        sigma_v (_type_): sigma_sph of the truncated SIS profile
        rc_kpc (_type_): truncation radius in kpc
        r_sym (_type_): coordinate along the symmetry axis
        qintr (_type_): intrinsic axis ratio. If oblate, qintr < 1; if prolate, qintr > 1
        plot_mge (int, optional): _description_. Defaults to 0.
        fignum (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: amplitude in M_sun/pc^3
        _type_: dispersion in pc
    """
    sis_profile = SIS_truncated_physical(sigma_v=sigma_v, rc = rc_kpc)
    intr_mge = Intr_MGE(profile=sis_profile, qintr=qintr, r_sym=r_sym)
    peak, sigma = intr_mge.MGE_param(kwargs_mge={'ngauss': 20, 'inner_slope': 3, 'outer_slope':1}, plot_mge=plot_mge, fignum=fignum)

    peak = peak / 1e9 # convert to [M_sun/pc^3]
    sigma = sigma * 1e3 # convert to [pc]

    return peak, sigma


def get_hernquist_intr_mge(m_star, Rs_kpc, qintr, plot_mge=0, fignum=1):

    hernquist_profile = Hernquist(Rs=Rs_kpc, sigma0=1e7)

    r_sym = np.geomspace(0.001, 10 * Rs_kpc, 300)

    intr_mge = Intr_MGE(profile=hernquist_profile, qintr=qintr, r_sym=r_sym)

    peak, sigma = intr_mge.MGE_param(kwargs_mge = {'ngauss': 20}, plot_mge=plot_mge, fignum=fignum)

    peak = peak / 1e9 # convert to [M_sun/pc^3]
    sigma = sigma * 1e3 # convert to [pc]

    mtot = intr_mge.MGE_mass_sph()
    peak = m_star / mtot * peak # rescale to desired input stellar mass

    return peak, sigma


def get_proj_mge(mge_proj, distance, inc):
    """get projected MGE

    Args:
        mge_proj (_type_): MGE_proj instance
        distance (_type_): angular diameter distance in Mpc, used to convert the dispersion to arcsec
        inc (_type_): inclination angle [rad]

    Returns:
        _type_: peak of the projected MGE [M_sun/pc^3]
        _type_: sigma of the projected MGE [arcsec]
        _type_: projected axis ratio 
    """
    surf = mge_proj.surf(inc=inc)
    qobs = mge_proj.qobs(inc=inc)

    qobs = np.full_like(surf, qobs)

    pc = distance * np.pi / 0.648
    sigma_intr = mge_proj.sigma
    sigma = sigma_intr / pc

    return surf, sigma, qobs

def get_sigma_e(surf_lum, sigma_lum, qobs_lum, jam, xbin, ybin):

    ifu_dim = int(np.sqrt(len(xbin)))
    if np.all(qobs_lum <= 1):
        flux = jam.flux
        reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(surf_lum, sigma_lum, qobs_lum)
    elif np.all(qobs_lum > 1):
        flux = np.reshape(jam.flux, (ifu_dim, ifu_dim)).T 
        flux = flux.flatten()
        reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(surf_lum, sigma_lum, 1/qobs_lum)
    else:
        raise ValueError('Apparent axis ratio must be constant with radius!')

    w = xbin**2 + (ybin/(1 - eps_e))**2 < reff_maj**2

    model = jam.model

    sig_e = np.sqrt((flux[w]*model[w]**2).sum()/flux[w].sum())

    print('sigma_e = {:.2f} km/s' .format(sig_e))

    return sig_e

def total_mass_3d(peak, sigma, qintr):
    return np.sum(peak * (sigma * np.sqrt(2 * np.pi))**3 * qintr)

def total_mass_2d(peak, sigma, qobs):
    return np.sum(2 * np.pi * peak * qobs * sigma**2)

def sphericalize_2d(sigma_proj, qobs):
    return sigma_proj * np.sqrt(qobs)

def get_sigma_e_dist(num, beta_const, prolate = False):

    sigma_e_axi_all = np.zeros(num)
    sigma_e_sph_all = np.zeros(num)

    mu_qintr_sr, sigma_qintr_sr = 0.74, 0.08 # intrinsic axis ratio prior from Li et al 2018
    qintr_all = qintr_prior(mu_qintr_sr, sigma_qintr_sr, num)
    inc_all = isotropic_inc(num)
    qobs_model_all = qobs_model(qintr_all, inc_all)

    if prolate:
        qintr_all = 1/qintr_all
        qobs_model_all = 1/qobs_model_all

    sigma_v = 200
    lens_cosmo = get_default_lens_cosmo()
    distance = lens_cosmo.Dd
    theta_sis = lens_cosmo.sis_sigma_v2theta_E(sigma_v)
    rc_kpc = lens_cosmo.arcsec2Mpc_lens(theta_sis) * 1e3 * 200
    r_sym = np.geomspace(0.01, 5 * rc_kpc, 300)

    m_star = 1e11
    rs_kpc = 8

    xx = np.linspace(-4 * theta_sis, 4 * theta_sis, 100)  # avoid (x,y)=(0,0)
    xbin, ybin = map(np.ravel, np.meshgrid(xx, xx))

    for i in range(num):

        qintr = qintr_all[i]
        inc = inc_all[i]
        qobs_model_i = qobs_model_all[i]

        # intrinsic density mge and projection
        peak_den_3d, sigma_den_3d = get_truncsis_intr_mge(sigma_v, rc_kpc, r_sym, qintr) # get intrinsic MGE
        mge_proj_den = MGE_Proj(peak_den_3d, sigma_den_3d, qintr)
        peak_den_2d, sigma_den_2d, qobs_den = get_proj_mge(mge_proj_den, distance, inc) # project MGE
        sigma_den_2d_sph = sphericalize_2d(sigma_den_2d, qobs_model_i) # sphericalize in 2d

        # intrinsic light mge and projection
        peak_lum_3d, sigma_lum_3d = get_hernquist_intr_mge(m_star, rs_kpc, qintr)
        mge_proj_lum = MGE_Proj(peak_lum_3d, sigma_lum_3d, qintr)
        peak_lum_2d, sigma_lum_2d, qobs_lum = get_proj_mge(mge_proj_lum, distance, inc)
        sigma_lum_2d_sph = sphericalize_2d(sigma_lum_2d, qobs_model_i) # sphericalize in 2d

        beta = np.ones_like(peak_lum_3d) * beta_const

        # axisymmetric 
        jam = jam_axi_proj(peak_lum_2d, sigma_lum_2d, qobs_lum, peak_den_2d, sigma_den_2d, qobs_den, np.degrees(inc), 0, distance, xbin, ybin, align='sph', beta=beta)  
        sigma_e_axi = get_sigma_e(peak_lum_2d, sigma_lum_2d, qobs_lum, jam, xbin, ybin)
        sigma_e_axi_all[i] = sigma_e_axi

        # sphericalized in 2d
        jam_sph = jam_axi_proj(peak_lum_2d, sigma_lum_2d_sph, np.ones_like(peak_lum_2d), peak_den_2d, sigma_den_2d_sph, np.ones_like(peak_den_2d), np.degrees(inc), 0, distance, xbin, ybin, align='sph', beta=beta) 
        sigma_e_sph = get_sigma_e(peak_lum_2d, sigma_lum_2d_sph, np.ones_like(peak_lum_2d), jam_sph, xbin, ybin)
        sigma_e_sph_all[i] = sigma_e_sph

    return np.vstack([sigma_e_axi_all, sigma_e_sph_all, inc_all, qintr_all, qobs_model_all])


num_proj = 2500
beta_const_list = [-0.2, 0, 0.2]
run_many_proj = True
prolate = True
if prolate:
    prolate_name = 'prolate'
else:
    prolate_name = ''

if run_many_proj:
    for i, beta in enumerate(beta_const_list):
        data = get_sigma_e_dist(num_proj, beta, prolate)
        np.save('./beta_{}_' .format(i) + prolate_name + '.npy' , data)
