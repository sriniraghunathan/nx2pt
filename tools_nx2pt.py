import numpy as np, os, sys, glob
import scipy as sc
import scipy.integrate as integrate
from pylab import *

def process_cosmo_param_dict_and_set_cosmo(cosmo_param_dict):
    import pyccl as ccl
    if cosmo_param_dict is None:
        cosmo_param_dict = {'omch2':0.1212, 'ombh2': 0.0202, 'h': 0.67, 'sigma8':0.81, 'ns': 0.96, 'w0': -1, 'wa': 0., 'mnu': 0.06}
    if 'ombh2' in cosmo_param_dict:
        omegab = cosmo_param_dict['ombh2']/cosmo_param_dict['h']**2.
    elif 'omegab' in cosmo_param_dict:
        omegab = cosmo_param_dict['omegab']
    else:
        print('Either ombh2 or omegab must exist.'); sys.exit()
    if 'omch2' in cosmo_param_dict:
        omegac = cosmo_param_dict['omch2']/cosmo_param_dict['h']**2.
    elif 'omegam' in cosmo_param_dict:
        omegac = cosmo_param_dict['omegam'] - omegab
    else:
        print('Either omch2 or omegam must exist.'); sys.exit()

    cosmo_param_dict['omegac'] = omegac
    cosmo_param_dict['omegab'] = omegab

    #cosmo = ccl.Cosmology(Omega_c=cosmo_param_dict['omegac'], Omega_b=cosmo_param_dict['omegab'], h=cosmo_param_dict['h'], sigma8=cosmo_param_dict['sigma8'], n_s=cosmo_param_dict['ns'])
    cosmo = ccl.Cosmology(Omega_c=cosmo_param_dict['omegac'], 
        Omega_b=cosmo_param_dict['omegab'], 
        h=cosmo_param_dict['h'], 
        sigma8=cosmo_param_dict['sigma8'], 
        n_s=cosmo_param_dict['ns'], 
        w0=cosmo_param_dict['w0'], wa=cosmo_param_dict['wa'], 
        m_nu=cosmo_param_dict['mnu'],
        transfer_function='boltzmann_camb',
        extra_parameters={'camb': {'dark_energy_model': 'ppf', 'kmax': cosmo_param_dict['kmax']}},        
        ##extra_parameters={'camb': {'dark_energy_model': 'ppf'}},
        )    
    return cosmo_param_dict, cosmo

def ngal_experiment(experiment, units = 'sr'):
    if experiment in ['roman_hls_lens']:
        ngal_per_arcmin2 = 51 #Page 3 of https://arxiv.org/pdf/2112.07681
    elif experiment in ['roman_hls_source']:
        ngal_per_arcmin2 = 66 #Page 3 of https://arxiv.org/pdf/2112.07681
    elif experiment in ['lsst_y1_lens']:
        ngal_per_arcmin2 = 18 #https://binny-org.github.io/binny/main/examples/survey_presets.html
    elif experiment in ['lsst_y1_source']:
        ngal_per_arcmin2 = 10 #https://binny-org.github.io/binny/main/examples/survey_presets.html
    elif experiment in ['lsst_y10_lens']:
        ngal_per_arcmin2 = 48 #https://binny-org.github.io/binny/main/examples/survey_presets.html
    elif experiment in ['lsst_y10_source']:
        ngal_per_arcmin2 = 27 #https://binny-org.github.io/binny/main/examples/survey_presets.html
    elif experiment in ['euclid_source', 'euclid_lens']:
        ngal_per_arcmin2 = 30.0 #https://binny-org.github.io/binny/main/examples/survey_presets.html and https://arxiv.org/pdf/2106.05267
    elif experiment in ['euclid_spec_source', 'euclid_spec_lens']:
        ngal_per_arcmin2 = 0.6 #https://arxiv.org/pdf/2106.05267

    ngal_per_sr = ngal_per_arcmin2 / np.radians(1./60.)**2.
    if units == 'sr':
        return ngal_per_sr
    elif units == 'arcmin2':
        return ngal_per_arcmin2

def shear_noise_experiment(experiment, els = None):
    if experiment in ['roman_hls_lens']:
        sigma_shear = 0.37 #Table 1 of https://arxiv.org/pdf/2112.07681
    elif experiment in ['lsst_y1_lens', 'lsst_y10_lens']:
        sigma_shear = 0.26
    elif experiment in ['euclid_source', 'euclid_lens']:
        sigma_shear = 0.26
    elif experiment in ['euclid_spec_source', 'euclid_spec_lens']:
        sigma_shear = 0.26

    ngal_per_sr = ngal_experiment(experiment)
    shape_noise = sigma_shear**2./ngal_per_sr
    if els is not None:
        shape_noise = np.tile(shape_noise, len(els))
    return shape_noise

def ngal_in_zbin(reqd_zbin, units = 'sr'):
    ngal_dic = {'zbin1': 1.78, 'zbin2': 1.77, 'zbin3': 1.78, 'zbin4': 1.78, 'zbin5': 1.78} #per arcmin^2
    ngal_per_arcmin2 = ngal_dic[reqd_zbin]
    ngal_per_sr = ngal_per_arcmin2 / np.radians(1./60.)**2.
    if units == 'sr':
        return ngal_per_sr
    elif units == 'arcmin2':
        return ngal_per_arcmin2

def get_shot_noise(experiment = None, reqd_zbin = None, els = None):
    if experiment is not None:
        ngal_per_sr = ngal_experiment(experiment)
    elif reqd_zbin is not None:
        ngal_per_sr = ngal_in_zbin(reqd_zbin)
    shot_noise = 1./ngal_per_sr
    if els is not None:
        shot_noise = np.tile(shot_noise, len(els))
    return shot_noise

def get_shape_noise(experiment = None, reqd_zbin = None, els = None):
    if experiment is not None:
        ngal_per_sr = ngal_experiment(experiment)
    elif reqd_zbin is not None:
        ngal_per_sr = ngal_in_zbin(reqd_zbin)
    shot_noise = 1./ngal_per_sr
    if els is not None:
        shot_noise = np.tile(shot_noise, len(els))
    return shot_noise    

def get_dN_dz(experiment, z, z0, beta, alpha = 2.):
    if experiment in ['euclid_source', 'euclid_lens', 'euclid_spec_source', 'roman_hls_source', 'roman_hls_lens', 'lsst_y1_source', 'lsst_y1_lens']:
        f_z = (z/z0)**alpha
    exp_val = -(z/z0)**beta
    return f_z * np.exp( exp_val )

def get_dngal_dz_photoz(experiment, zbin, z1 = 0.01, z2 = 4., total_bins = 500, howmanysigmasforint = 1, cosmo_param_dict = None):
    zbin_centre = np.mean(zbin)
    zarr = np.linspace(z1, z2, total_bins)
    if experiment == 'euclid_source':
        z0 = 0.64
        beta = 3/2
        alpha = 2.
        sigma_z = 0.05 * (1+zbin_centre)
        zbincntr = np.arange(len(zarr)) + 1
        bias_model = 1.2 + (zbincntr * 0.1)
    elif experiment == 'euclid_lens':
        z0 = 1.
        beta = 3/2
        alpha = 2.
        sigma_z = 0.05 * (1+zbin_centre)
        zbincntr = np.arange(len(zarr)) + 1
        bias_model = 1.2 + (zbincntr * 0.1)
    elif experiment == 'euclid_spec_source':
        z0 = 0.64
        beta = 3/2
        alpha = 2.
        sigma_z = 0.001 * (1+zbin_centre)
        zbincntr = np.arange(len(zbin)) + 1
        bias_model = 1.2 + (zbincntr * 0.1)
    elif experiment == 'roman_hls_source':
        z0 = 0.13
        beta = 0.66
        alpha = 2.
        sigma_z = 0.05 * (1+zbin_centre)
        zbincntr = np.arange(len(zarr)) + 1
        bias_model = 1.2 + (zbincntr * 0.1)
    elif experiment == 'roman_hls_lens':
        """
        z0 = 0.08
        beta = 0.58
        alpha =2.
        """
        z0 = 0.8
        beta = 0.58
        alpha = 2.
        sigma_z = 0.05 * (1+zbin_centre)
        zbincntr = np.arange(len(zarr)) + 1
        bias_model = 1.2 + (zbincntr * 0.1)
    elif experiment in ['lsst_y1_source', 'lsst_y1_lens']:
        import pyccl as ccl
        cosmo_param_dict, cosmo = process_cosmo_param_dict_and_set_cosmo(cosmo_param_dict)        
        scale_fac_arr = 1/(1+zarr)
        #D_a = ccl.background.angular_diameter_distance(cosmo, scale_fac_arr)
        Dz = ccl.growth_factor(cosmo, scale_fac_arr)
        if experiment == 'lsst_y1_source':
            z0 = 0.28
            alpha = 2.
            beta = 0.94
            sigma_z = 0.05 * (1+zbin_centre)
        elif experiment == 'lsst_y1_lens':
            z0 = 0.13
            alpha = 2.
            beta = 0.78
            sigma_z = 0.03 * (1+zbin_centre)
        zbincntr = np.arange(len(zarr)) + 1
        bias_model = 0.9/Dz
        bias_model[np.isinf(bias_model) | np.isnan(bias_model)] = 0.

    #zmin, zmax = zbin - (howmanysigmasforint * sigma_z/2), zbin + (howmanysigmasforint * sigma_z/2)
    zmin, zmax = zbin
    dndz = get_dN_dz(experiment, zarr, z0, beta, alpha)
    dndz_bin = 0.5 * dndz * ( sc.special.erf( (zmax-zarr) / np.sqrt(2) / sigma_z ) - sc.special.erf( (zmin-zarr) / np.sqrt(2) / sigma_z )  )
    exp_specs_dic = {'z0': z0, 'alpha': alpha, 'beta': beta}
    return zarr, dndz_bin, bias_model, exp_specs_dic

'''
def get_dngal_dz_photoz_full(experiment, zbin, z1 = 0, z2 = 20., zdelta = 0.01, nz = 1000, howmanysigmasforint = 1):
    import scipy as sc
    import scipy.integrate as integrate
    zarr = np.arange(z1, z2, zdelta)
    if experiment == 'euclid':
        z0 = 0.64
        beta = 3/2
        sigma_z = 0.05 * (1+zbin)
    
    dndz = get_dN_dz(experiment, zarr, z0, beta)
    def photo_z_error_int(zforint):
        """ 
        Returns the photo-z error integrated PDF.
        """
        exp_val = -1/2. * ( ( zforint - zbin )/sigma_z )**2.
        p_z = 1/np.sqrt( 2 * np.pi )/sigma_z * np.exp( exp_val )
        ##plot(zs, p_z); show(); sys.exit()
        return p_z

    zmin, zmax = zbin - (howmanysigmasforint * sigma_z/2), zbin + (howmanysigmasforint * sigma_z/2)
    zs   = np.linspace(zmin, zmax, nz)
    p_z = integrate.simps( photo_z_error_int(zs), x=zs )
    #print(zbin, sigma_z, p_z); sys.exit()
    dndz_bin = dndz * p_z

    dndz_bin = 0.5 * dndz * ( sc.special.erf( (zmax-zarr) / np.sqrt(2) / sigma_z ) - sc.special.erf( (zmin-zarr) / np.sqrt(2) / sigma_z )  )
    return zarr, dndz_bin
'''

def get_3x2pt_lsst_data(which_year = 'y1', which_set = 'all', data_or_dervitive = 'data', param_for_derivative = None, obtain_cov = False, lsst_fsky_dic = None):
    assert which_set in ['all', 'cosmo']
    if lsst_fsky_dic is None:
        lsst_fsky_dic = {'cl_s_s': 0.4, 'cl_g_s': 0.4, 'cl_g_g': 0.4}
    lsstfd = 'data/LSST_3x2pt/srd_comsology/'
    if data_or_dervitive == 'data':
        lsstfname = '%s/data_vectors/cmb_%s_3x2pt_data_vector_%s_priors.npy' %(lsstfd, which_year, which_set)
    elif data_or_dervitive == 'derivative':
        assert param_for_derivative is not None
        lsstfname = '%s/derivatives/cmb_%s_3x2pt_%s_derivative_%s_priors.npy' %(lsstfd, which_year, param_for_derivative, which_set)
    lsstdata = np.load( lsstfname, allow_pickle=True )

    ell = np.geomspace(2, 2000, 20)
    reclen = len(ell)

    if which_year == 'y1':
        ss_len, gs_len, gg_len = 15, 7, 5
    elif which_year == 'y10':
        ss_len, gs_len, gg_len = 15, 25, 10

    ###############
    #data
    ###############
    #shear
    cl_s_s = lsstdata[:ss_len, :]
    total_s_s_zbins = 5
    zbin_s_s_arr = np.arange( total_s_s_zbins )
    s_s_zbin_combs = [(z1, z2) for z1 in zbin_s_s_arr for z2 in zbin_s_s_arr if z1<=z2]
    ###print(s_s_zbin_combs); ##sys.exit()

    #clustering
    cl_g_g = lsstdata[ss_len+gs_len: , :]
    total_g_g_zbins = 5
    zbin_s_s_arr = np.arange( total_g_g_zbins )
    g_g_zbin_combs = [(z, z) for z in zbin_s_s_arr] #only autos
    ###print(g_g_zbin_combs); ##sys.exit()

    #GGL
    cl_g_s = lsstdata[ss_len: ss_len+gs_len, :]
    total_g_g_zbins = 5
    zbin_s_s_arr = np.arange( total_g_g_zbins )
    """
    g_s_zbin_combs = [(z1, z2) for z1 in zbin_s_s_arr for z2 in zbin_s_s_arr if z1<=z2]
    excluded_pairs = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3)] #from binny
    #g_s_zbin_combs_ref = [z1z2 if z1z2 not in excluded_pairs else None for z1z2 in g_s_zbin_combs]
    g_s_zbin_combs_ref = [z1z2 for z1z2 in g_s_zbin_combs if z1z2 not in excluded_pairs]
    g_s_zbin_combs = g_s_zbin_combs_ref
    """
    g_s_zbin_combs = [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)] #from binny
    ###print(g_s_zbin_combs)
    """
    for z1z2 in g_s_zbin_combs:
        if z1z2 in excluded_pairs: continue 
        print(z1z2)
    print(len(g_s_zbin_combs_ref)); 
    """

    """
    clf()
    color_arr = [cm.jet(int(d)) for d in np.linspace(0, 255, 10)]
    for iii in range(6):
        plot(cl_s_s[iii], color = color_arr[iii])
    show()
    sys.exit()
    """

    #initiate
    zbin_arr = np.copy(zbin_s_s_arr)
    all_zcombs = [(cntr1, cntr2) for cntr1 in range(len(zbin_arr)) for cntr2 in range(len(zbin_arr)) if cntr1<=cntr2]
    all_combs_ss = ['cl_s_s_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gg = ['cl_g_g_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gs = np.unique( ['cl_g_s_z%s_z%s' %(z1z2[0], z1z2p[1]) for z1z2 in all_zcombs for z1z2p in all_zcombs] ).tolist()

    data_vector_dic = {}
    data_vector_dic['cl_g_g'] = {}
    data_vector_dic['cl_s_s'] = {}
    data_vector_dic['cl_g_s'] = {}
    #ss
    for curr_comb in all_combs_ss:
        zbin1, zbin2 = curr_comb.split('_z')[1:]                
        zbin1, zbin2 = int(zbin1), int(zbin2)
        data_vector_dic['cl_s_s'][(zbin1, zbin2)] = np.zeros( reclen )
    #gg
    for curr_comb in all_combs_gg:
        zbin1, zbin2 = curr_comb.split('_z')[1:]                
        zbin1, zbin2 = int(zbin1), int(zbin2)
        data_vector_dic['cl_g_g'][(zbin1, zbin2)] = np.zeros( reclen )
    #gs
    for curr_comb in all_combs_gs:
        zbin1, zbin2 = curr_comb.split('_z')[1:]                
        zbin1, zbin2 = int(zbin1), int(zbin2)
        data_vector_dic['cl_g_s'][(zbin1, zbin2)] = np.zeros( reclen )

    #Fill shear
    if data_or_dervitive == 'data':
        nl_s_s = shear_noise_experiment(experiment = 'lsst_y1_lens')
    else:
        nl_s_s = np.zeros( reclen )
    for cntr, curr_comb  in enumerate( s_s_zbin_combs ):
        zbin1, zbin2 = zbin_arr[curr_comb[0]], zbin_arr[curr_comb[1]]
        data_vector_dic['cl_s_s'][(zbin1, zbin2)] = cl_s_s[cntr] + nl_s_s
        #data_vector_dic['cl_s_s'][(zbin2, zbin1)] = cl_s_s[cntr] + nl_s_s

    #Fill clustering
    if data_or_dervitive == 'data':
        nl_g_g = get_shot_noise(experiment = 'lsst_y1_source')
    else:
        nl_g_g = np.zeros( reclen )
    for cntr, curr_comb  in enumerate( g_g_zbin_combs ):
        zbin1, zbin2 = zbin_arr[curr_comb[0]], zbin_arr[curr_comb[1]]
        data_vector_dic['cl_g_g'][(zbin1, zbin2)] = cl_g_g[cntr] + nl_g_g
        #data_vector_dic['cl_g_g'][(zbin2, zbin1)] = cl_g_g[cntr] + nl_g_g

    #Fill GGL
    for cntr, curr_comb  in enumerate( g_s_zbin_combs ):
        zbin1, zbin2 = zbin_arr[curr_comb[0]], zbin_arr[curr_comb[1]]
        data_vector_dic['cl_g_s'][(zbin1, zbin2)] = cl_g_s[cntr]
        #data_vector_dic['cl_g_s'][(zbin2, zbin1)] = cl_g_s[cntr]

    ###print( len(data_vector_dic['cl_s_s']), len(data_vector_dic['cl_g_g']), len(data_vector_dic['cl_g_s']), 'hi' ); sys.exit()

    if (0):###data_or_dervitive == 'data': #show plot
        clf()
        ax = subplot(111, xscale = 'log', yscale = 'log')
        dl_fac = ell * (ell+1)/2/np.pi
        cmap = cm.RdYlBu_r
        color_arr = [cmap(int(d)) for d in np.linspace(0, 255, len(s_s_zbin_combs))]
        for cntr, curr_comb  in enumerate( s_s_zbin_combs ):
            plot( ell, dl_fac*cl_s_s[cntr], color = color_arr[cntr], label = r'%s x %s' %(curr_comb[0], curr_comb[1]))
        plot( ell, dl_fac*nl_s_s, color = 'black', ls = '--', lw = 2., label = r'Noise: Shear')
        legend(loc = 1, fontsize = 10, ncol = 2)
        xlabel(r'Multipole $\ell$', fontsize = 14)
        ylabel(r'$C_{\ell}^{\gamma \gamma}$', fontsize = 14)
        show(); 
            
        clf()
        ax = subplot(111, xscale = 'log', yscale = 'log')
        cmap = cm.RdYlBu_r
        color_arr = [cmap(int(d)) for d in np.linspace(0, 255, len(s_s_zbin_combs))]
        for cntr, curr_comb  in enumerate( g_g_zbin_combs ):
            plot( ell, dl_fac*cl_g_g[cntr], color = color_arr[cntr], label = r'%s x %s' %(curr_comb[0], curr_comb[1]))
        plot( ell, dl_fac*nl_g_g, color = 'black', ls = '--', lw = 2., label = r'Noise: Shear')
        legend(loc = 1, fontsize = 10, ncol = 2)
        xlabel(r'Multipole $\ell$', fontsize = 14)
        ylabel(r'$C_{\ell}^{gg}$', fontsize = 14)
        show(); sys.exit()            
    
    if not obtain_cov:
        return ell, zbin_arr, data_vector_dic
    else:
        #get cov
        full_cov_mat_dic = get_nx2t_cov(ell, data_vector_dic, zbin_arr, lsst_fsky_dic)
        return ell, zbin_arr, data_vector_dic, full_cov_mat_dic

def get_ia_nla_model(zarr, a_ia, eta_ia, exp_specs_dic):
    ia_nla_model = a_ia * ((1.0 + zarr) / (1.0 + exp_specs_dic['z0'])) ** eta_ia
    return ia_nla_model

def get_scale_dependent_bias(cosmo_param_dict, zarr, bias_model, exp_specs_dic, delta_c = 1.686, c = 1., kmin = 1e-3, kmax = 1., kdelta = 100, kpivot = 0.05):
    assert 'fnl_loc' in cosmo_param_dict
    import pyccl as ccl
    cosmo_param_dict, cosmo = process_cosmo_param_dict_and_set_cosmo(cosmo_param_dict)
    scale_fac_arr = 1/(1+zarr)
    Dz = ccl.growth_factor(cosmo, scale_fac_arr)

    karr = np.geomspace(kmin, kmax, kdelta)
    pk_lin = ccl.linear_matter_power(cosmo, karr, scale_fac_arr)
    p_primordial = (2 * np.pi**2 / karr**3) * cosmo_param_dict['As'] * (karr / kpivot)**(cosmo_param_dict['ns'] - 1.0)
    transfer_k = (1./Dz[:,np.newaxis]) * np.sqrt(pk_lin[np.newaxis, :] / p_primordial)
    
    t1 = (bias_model-1) * cosmo_param_dict['fnl_loc'] * delta_c
    t2 = 3 * cosmo_param_dict['omegam'] * (100*cosmo_param_dict['h'])**2.
    t3 = karr**2. * transfer_k * Dz[:,np.newaxis]
    scale_dep_bias_model = (t1[:,np.newaxis] * t2)/t3

    
    if experiment == 'lsst_y1_source':
        z0 = 0.28
        alpha = 2.
        beta = 0.94
        sigma_z = 0.05 * (1+zbin_centre)
    elif experiment == 'lsst_y1_lens':
        z0 = 0.13
        alpha = 2.
        beta = 0.78
        sigma_z = 0.03 * (1+zbin_centre)
    zbincntr = np.arange(len(zarr)) + 1
    bias_model = 0.9/Dz
    bias_model[np.isinf(bias_model) | np.isnan(bias_model)] = 0.

def get_combined_experiment_dndz(combined_exp, zmin, zmax, zbinwidth, cosmo_param_dict, delimited = '+++++'): 
    combined_experiment_arr = combined_exp.split(delimited)
    zbin_arr = [(z, z+zbinwidth) for z in np.arange(zmin, zmax, zbinwidth)]
    dndz_lensgalsforclus_dic_combexp = {}
    dndz_sourcegalsforshear_dic_combexp = {}
    zbin_mid_arr = []
    total_cntr = 0
    for curr_exp_cntr, curr_exp in enumerate( combined_experiment_arr ):
        for cntr, zbin in enumerate( zbin_arr ):
            zbin_mid = (zbin[0] + zbin[1] )/2.
            zarr_source_combexp, dndz_bin_source_combexp, bias_source_combexp, exp_specs_dic_source_combexp = get_dngal_dz_photoz('%s_source' %(curr_exp), zbin, cosmo_param_dict = cosmo_param_dict)
            dndz_lensgalsforclus_dic_combexp[total_cntr] = [zarr_source_combexp, dndz_bin_source_combexp, bias_source_combexp, exp_specs_dic_source_combexp]
            zarr_lens_combexp, dndz_bin_lens_combexp, bias_lens_combexp, exp_specs_dic_lens_combexp = get_dngal_dz_photoz('%s_lens' %(curr_exp), zbin, cosmo_param_dict = cosmo_param_dict)
            dndz_sourcegalsforshear_dic_combexp[total_cntr] = [zarr_lens_combexp, dndz_bin_lens_combexp, bias_lens_combexp, exp_specs_dic_lens_combexp]
            zbin_mid_arr.append( total_cntr )
            total_cntr += 1
    return zbin_mid_arr, dndz_lensgalsforclus_dic_combexp, dndz_sourcegalsforshear_dic_combexp

def get_nx2pt_data_vectors_and_cov(experiment, 
    zmin = 0.01, zmax = 4.1, zbinwidth = 0.5, 
    ell = None, 
    cosmo_param_dict = None, 
    has_rsd = False, 
    z_lss = 1100., 
    obtain_cov = False, 
    include_ia_errors = True,
    add_scale_dependent_bias = False,
    include_cmb_lensing = False, 
    fsky_dic = None, 
    cmb_experiment_for_kappa = 'advanced_sobaseline',
    show_plot = False,
    zbin_mid_arr = None, 
    dndz_lensgalsforclus_dic = None, 
    dndz_sourcegalsforshear_dic = None, 
    ):

    import pyccl as ccl
    if ell is None:
        ell = np.geomspace(2, 2000, 20)
    cosmo_param_dict, cosmo = process_cosmo_param_dict_and_set_cosmo(cosmo_param_dict)
    ###print(cosmo); sys.exit()

    #get dN/dz
    if dndz_lensgalsforclus_dic is None:

        print('#get dN/dz')
        zbin_arr = [(z, z+zbinwidth) for z in np.arange(zmin, zmax, zbinwidth)]
        ###print('Total z-bins = %s' %(len(zbin_arr)));### sys.exit()

        if show_plot:
            color_arr = [cm.Reds_r(int(d)) for d in np.linspace(20, 255, len(zbin_arr))]
            color_arr_v2 = [cm.Greens_r(int(d)) for d in np.linspace(20, 255, len(zbin_arr))]
        

        dndz_lensgalsforclus_dic = {}
        dndz_sourcegalsforshear_dic = {}
        zbin_mid_arr = []
        for cntr, zbin in enumerate( zbin_arr ):
            zbin_mid = (zbin[0] + zbin[1] )/2.
            zarr_source, dndz_bin_source, bias_source, exp_specs_dic_source = get_dngal_dz_photoz('%s_source' %(experiment), zbin, cosmo_param_dict = cosmo_param_dict)
            dndz_lensgalsforclus_dic[zbin_mid] = [zarr_source, dndz_bin_source, bias_source, exp_specs_dic_source]
            zarr_lens, dndz_bin_lens, bias_lens, exp_specs_dic_lens = get_dngal_dz_photoz('%s_lens' %(experiment), zbin, cosmo_param_dict = cosmo_param_dict)
            dndz_sourcegalsforshear_dic[zbin_mid] = [zarr_lens, dndz_bin_lens, bias_lens, exp_specs_dic_lens]
            zbin_mid_arr.append( zbin_mid )

        if show_plot:
            for cntr, zbin in enumerate( zbin_arr ):
                zarr_lens, dndz_bin_lens, bias_lens, exp_specs_dic_lens = dndz_sourcegalsforshear_dic[zbin_mid]
                plot( zarr_lens, dndz_bin_lens, color = color_arr_v2[cntr], ls = '-.')
                zarr_source, dndz_bin_source, bias_source, exp_specs_dic_source = dndz_lensgalsforclus_dic[zbin_mid]
                plot( zarr_source, dndz_bin_source, color = color_arr[cntr])
            xlim(0., 4.)
            plot([], [], 'k-', label = r'Source galaxies')
            plot([], [], 'k-.', label = r'Lens galaxies')
            xlabel(r'Redshift $z$', fontsize = fsval); ylabel(r'$dN/dz$', fontsize = fsval)
            legend(loc = 1, fontsize = fsval-2); show()
    
    #data vector dict
    print('#data vector dict'); ###sys.exit()
    data_vector_dic = {}
    noise_vector_dic = {}
    if include_cmb_lensing:
        #CMB lensing tracer
        curr_t_k = ccl.CMBLensingTracer(cosmo, z_source=z_lss)
    
        #kk - CMB lensing convergence
        curr_cl_k_k = ccl.angular_cl( cosmo, curr_t_k, curr_t_k, ell=ell)
        data_vector_dic['cl_k_k'] = {(0, 0): curr_cl_k_k}
        data_vector_dic['cl_k_g'] = {}
        data_vector_dic['cl_k_s'] = {}

        #CMB-lensing noise.
        cmb_lensing_noise_fname_searchstr = 'data/cmb/lensing/%s_*lmin300_lmax4000_lmaxtt3500.npy' %(cmb_experiment_for_kappa)
        cmb_lensing_noise_fname = glob.glob( cmb_lensing_noise_fname_searchstr )[0]
        ###print('\tcmb_lensing_noise_fname = %s' %(cmb_lensing_noise_fname))
        cmbkappadic = np.load(cmb_lensing_noise_fname, allow_pickle=True, encoding = 'latin1').item()
        tmpels, tmpnlkk = cmbkappadic['els'], cmbkappadic['Nl_MV'].real
        curr_nl_k_k = np.interp(ell, tmpels, tmpnlkk)
        """
        clf()
        ax = subplot(111, yscale = 'log')
        plot( ell, curr_cl_k_k )
        plot( ell, curr_nl_k_k )
        show(); sys.exit()
        """
        curr_nl_k_k = curr_nl_k_k
        noise_vector_dic['cl_k_k'] = {(0,0): curr_nl_k_k}

    data_vector_dic['cl_g_g'] = {}
    data_vector_dic['cl_s_s'] = {}
    data_vector_dic['cl_g_s'] = {}
    noise_vector_dic['cl_g_g'] = {}
    noise_vector_dic['cl_s_s'] = {}

    #shot and shape noise 
    nl_g_g = get_shot_noise(experiment = 'lsst_y1_source', els = ell)
    nl_s_s = shear_noise_experiment(experiment = 'lsst_y1_lens', els = ell)

    #all possible combinations
    all_zcombs = [(cntr1, cntr2) for cntr1 in range(len(zbin_mid_arr)) for cntr2 in range(len(zbin_mid_arr)) if cntr1<=cntr2]
    all_combs_ss = ['cl_s_s_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gg = ['cl_g_g_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gs = np.unique( ['cl_g_s_z%s_z%s' %(z1z2[0], z1z2p[1]) for z1z2 in all_zcombs for z1z2p in all_zcombs] ).tolist()
    ##print( len(all_combs_ss), len(all_combs_gg), len(all_combs_gs)); sys.exit()

    if include_cmb_lensing:
        for cntr1, zbin1 in enumerate( dndz_lensgalsforclus_dic ):
            zarr_lensgal1, dndz_lensgal1, bias_lensgal1, exp_specs_dic_lensgal1 = dndz_lensgalsforclus_dic[zbin1]
            zarr_sourcegal1, dndz_sourcegal1, bias_sourcegal1, exp_specs_dic_sourcegal1 = dndz_sourcegalsforshear_dic[zbin1]
            
            #tracers
            curr_t_g1 = ccl.NumberCountsTracer(cosmo, has_rsd=has_rsd, dndz=(zarr_lensgal1, dndz_lensgal1), bias=(zarr_lensgal1, bias_lensgal1))
            curr_t_s1 = ccl.WeakLensingTracer(cosmo, has_shear=True, dndz=(zarr_sourcegal1, dndz_sourcegal1))
            if include_ia_errors:
                curr_s1_ia_nla_model = get_ia_nla_model(zarr_sourcegal1, cosmo_param_dict['a_ia'], cosmo_param_dict['eta_ia'], exp_specs_dic_sourcegal1)
                curr_s1_ia_bias_tuple = (zarr_sourcegal1, curr_s1_ia_nla_model)
                curr_t_s1_with_nla = ccl.WeakLensingTracer(cosmo, has_shear=True, dndz=(zarr_sourcegal1, dndz_sourcegal1), ia_bias=curr_s1_ia_bias_tuple, 
                    use_A_ia=True,  # Instructs CCL to scale the array via standard critical density
                )
                curr_t_s1 = curr_t_s1_with_nla

            #kappa_g
            curr_cl_k_g = ccl.angular_cl( cosmo, curr_t_k, curr_t_g1, ell=ell)
            #kappa_shear
            curr_cl_k_s = ccl.angular_cl( cosmo, curr_t_k, curr_t_s1, ell=ell)
            data_vector_dic['cl_k_g'][(0, cntr1)] = curr_cl_k_g
            data_vector_dic['cl_k_s'][(0, cntr1)] = curr_cl_k_s
        # print(data_vector_dic['cl_k_k'])
        # print(data_vector_dic['cl_k_g'])
        # print(data_vector_dic['cl_k_s'])
        # sys.exit()

    #multiple zbins for lens/source gaalxies and their cross with CMB lensing
    zbin_arr = []
    for cntr1, zbin1 in enumerate( dndz_lensgalsforclus_dic ):
        zbin_arr.append(cntr1)
        zarr_lensgal1, dndz_lensgal1, bias_lensgal1, exp_specs_dic_lensgal1 = dndz_lensgalsforclus_dic[zbin1]
        zarr_sourcegal1, dndz_sourcegal1, bias_sourcegal1, exp_specs_dic_sourcegal1 = dndz_sourcegalsforshear_dic[zbin1]
        
        #tracers
        curr_t_g1 = ccl.NumberCountsTracer(cosmo, has_rsd=has_rsd, dndz=(zarr_lensgal1, dndz_lensgal1), bias=(zarr_lensgal1, bias_lensgal1))
        curr_t_s1 = ccl.WeakLensingTracer(cosmo, has_shear=True, dndz=(zarr_sourcegal1, dndz_sourcegal1))
        if include_ia_errors:
            curr_s1_ia_nla_model = get_ia_nla_model(zarr_sourcegal1, cosmo_param_dict['a_ia'], cosmo_param_dict['eta_ia'], exp_specs_dic_sourcegal1)
            curr_s1_ia_bias_tuple = (zarr_sourcegal1, curr_s1_ia_nla_model)
            curr_t_s1_with_nla = ccl.WeakLensingTracer(cosmo, has_shear=True, dndz=(zarr_sourcegal1, dndz_sourcegal1), ia_bias=curr_s1_ia_bias_tuple, 
                use_A_ia=True,  # Instructs CCL to scale the array via standard critical density
            )
            curr_t_s1 = curr_t_s1_with_nla

        for cntr2, zbin2 in enumerate( dndz_sourcegalsforshear_dic ):

            zarr_lensgal2, dndz_lensgal2, bias_lensgal2, exp_specs_dic_lensgal2 = dndz_lensgalsforclus_dic[zbin2]
            zarr_sourcegal2, dndz_sourcegal2, bias_sourcegal2, exp_specs_dic_sourcegal2 = dndz_sourcegalsforshear_dic[zbin2]

            #tracers
            curr_t_g2 = ccl.NumberCountsTracer(cosmo, has_rsd=has_rsd, dndz=(zarr_lensgal2, dndz_lensgal2), bias=(zarr_lensgal2, bias_lensgal2))
            curr_t_s2 = ccl.WeakLensingTracer(cosmo, has_shear=True, dndz=(zarr_sourcegal2, dndz_sourcegal2))
            if include_ia_errors:
                curr_s2_ia_nla_model = get_ia_nla_model(zarr_sourcegal2, cosmo_param_dict['a_ia'], cosmo_param_dict['eta_ia'], exp_specs_dic_sourcegal2)
                curr_s2_ia_bias_tuple = (zarr_sourcegal2, curr_s2_ia_nla_model)
                curr_t_s2_with_nla = ccl.WeakLensingTracer(cosmo, has_shear=True, dndz=(zarr_sourcegal2, dndz_sourcegal2), ia_bias=curr_s2_ia_bias_tuple, 
                    use_A_ia=True,  # Instructs CCL to scale the array via standard critical density
                )
                curr_t_s2 = curr_t_s2_with_nla

            
            #gg - galaxy clustering
            curr_cl_g_g = ccl.angular_cl( cosmo, curr_t_g1, curr_t_g2, ell=ell)
            #ss - shear
            curr_cl_s_s = ccl.angular_cl( cosmo, curr_t_s1, curr_t_s2, ell=ell)
            #gs - galaxy-shear
            curr_cl_g_s = ccl.angular_cl( cosmo, curr_t_g1, curr_t_s2, ell=ell)
            if (0):##include_ia_errors:
                #curr_cl_s_s_with_ia_errors = ccl.angular_cl( cosmo, curr_t_s1_with_nla, curr_t_s2_with_nla, ell=ell)
                #curr_nl_s_s_ia_errors = curr_cl_s_s_with_ia_errors - curr_cl_s_s #Note that this can be negative.
                curr_cl_s_s = ccl.angular_cl( cosmo, curr_t_s1_with_nla, curr_t_s2_with_nla, ell=ell)
                ##print(curr_cl_s_s); sys.exit()

            tmp_cl_g_g_key = 'cl_g_g_z%s_z%s' %(cntr1, cntr2)
            tmp_cl_s_s_key = 'cl_s_s_z%s_z%s' %(cntr1, cntr2)
            tmp_cl_g_s_key = 'cl_g_s_z%s_z%s' %(cntr1, cntr2)
            if tmp_cl_g_g_key in all_combs_gg:
                data_vector_dic['cl_g_g'][(cntr1, cntr2)] = curr_cl_g_g # + nl_g_g
                noise_vector_dic['cl_g_g'][(cntr1, cntr2)] = nl_g_g
            if tmp_cl_s_s_key in all_combs_ss:
                data_vector_dic['cl_s_s'][(cntr1, cntr2)] = curr_cl_s_s # + nl_s_s# + curr_nl_s_s_ia_errors
                noise_vector_dic['cl_s_s'][(cntr1, cntr2)] = nl_s_s #+ curr_nl_s_s_ia_errors
            if tmp_cl_g_s_key in all_combs_gs:
                data_vector_dic['cl_g_s'][(cntr1, cntr2)] = curr_cl_g_s
        
    ###print( len(data_vector_dic['cl_s_s']), len(data_vector_dic['cl_g_g']), len(data_vector_dic['cl_g_s']) ); sys.exit()
    zbin_arr = np.asarray(zbin_arr)
    if not obtain_cov:
        return ell, zbin_arr, dndz_lensgalsforclus_dic, dndz_sourcegalsforshear_dic, data_vector_dic, noise_vector_dic
    else:
        assert fsky_dic is not None
        #get cov
        print('#get cov')
        full_cov_mat_dic = get_nx2t_cov(ell, data_vector_dic, zbin_arr, fsky_dic, noise_vector_dic = noise_vector_dic)
        return ell, zbin_arr, dndz_lensgalsforclus_dic, dndz_sourcegalsforshear_dic, data_vector_dic, noise_vector_dic, full_cov_mat_dic

def get_nx2pt_derivatives(experiment, 
    zmin, zmax, zbinwidth, 
    params_for_deriv, 
    param_dict, 
    ell = None, 
    has_rsd = False, 
    step_percent = 0.01, 
    data_vector_dic = None, 
    include_cmb_lensing = False,
    combined_exp_delimiter = '+++++',
    ):

    import copy
    #loop over parameters and get the observables/derivatives by modifying each param as param-step and param+step
    derivative_vector_dic = {}
    for ppp in params_for_deriv:
        pval = param_dict[ppp]
        if pval == 0.:
            pstepval = 0.01
        else:
            pstepval = pval * step_percent
        param_dict_low = copy.deepcopy(param_dict)
        param_dict_high = copy.deepcopy(param_dict)
        param_dict_low[ppp] = pval - pstepval
        param_dict_high[ppp] = pval + pstepval
        print(ppp, param_dict[ppp], param_dict_low[ppp], param_dict_high[ppp]);

        ##if ppp.find('b_')==-1 and ppp not in ['a_ia', 'eta_ia']: continue
        ##if ppp not in ['a_ia', 'eta_ia']: continue
        ##if ppp.find('b_')==-1: continue

        if ppp.find('b_')==0:
            if data_vector_dic is None:
                ell, zbin_arr, dndz_lensgalsforclus_dic, dndz_sourcegalsforshear_dic, data_vector_dic, noise_vector_dic = get_nx2pt_data_vectors_and_cov(experiment, zmin, zmax, zbinwidth, ell = ell, cosmo_param_dict = param_dict, include_cmb_lensing = include_cmb_lensing)
            data_vector_dic_low, data_vector_dic_high = {}, {}
            for obskey in data_vector_dic:
                data_vector_dic_low[obskey] = {}
                data_vector_dic_high[obskey] = {}
                for z1z2 in data_vector_dic[obskey]:
                    p1forz1bin = 'b_%s' %(z1z2[0]+1)
                    p2forz2bin = 'b_%s' %(z1z2[0]+1)
                    if obskey == 'cl_g_g' and ppp in [p1forz1bin, p2forz2bin]:
                        data_vector_dic_low[obskey][z1z2] = param_dict_low[ppp] * data_vector_dic[obskey][z1z2]
                        data_vector_dic_high[obskey][z1z2] = param_dict_high[ppp] * data_vector_dic[obskey][z1z2]
                    else:
                        data_vector_dic_low[obskey][z1z2] = data_vector_dic[obskey][z1z2]
                        data_vector_dic_high[obskey][z1z2] = data_vector_dic[obskey][z1z2]
        else:
            if experiment.find(combined_exp_delimiter)==-1:
                ell, zbin_arr_low, dndz_lensgalsforclus_dic_low, dndz_sourcegalsforshear_dic_low, data_vector_dic_low, noise_vector_dic_low = get_nx2pt_data_vectors_and_cov(experiment, zmin, zmax, zbinwidth, ell = ell, cosmo_param_dict = param_dict_low, include_cmb_lensing = include_cmb_lensing)
                ell, zbin_arr_high, dndz_lensgalsforclus_dic_high, dndz_sourcegalsforshear_dic_high, data_vector_dic_high, noise_vector_dic_high = get_nx2pt_data_vectors_and_cov(experiment, zmin, zmax, zbinwidth, ell = ell, cosmo_param_dict = param_dict_high, include_cmb_lensing = include_cmb_lensing)
            else:
                zbin_mid_arr_low, dndz_lensgalsforclus_dic_low, dndz_sourcegalsforshear_dic_low = get_combined_experiment_dndz(experiment, zmin, zmax, zbinwidth, param_dict_low)
                ell, zbin_arr_low, dndz_lensgalsforclus_dic_low, dndz_sourcegalsforshear_dic_low, data_vector_dic_low, noise_vector_dic_low = get_nx2pt_data_vectors_and_cov(experiment, zmin, zmax, zbinwidth, ell = ell, cosmo_param_dict = param_dict_low, include_cmb_lensing = include_cmb_lensing, zbin_mid_arr = zbin_mid_arr_low, dndz_lensgalsforclus_dic = dndz_lensgalsforclus_dic_low, dndz_sourcegalsforshear_dic = dndz_sourcegalsforshear_dic_low)
                
                zbin_mid_arr_high, dndz_lensgalsforclus_dic_high, dndz_sourcegalsforshear_dic_high = get_combined_experiment_dndz(experiment, zmin, zmax, zbinwidth, param_dict_high)
                ell, zbin_arr_high, dndz_lensgalsforclus_dic_high, dndz_sourcegalsforshear_dic_high, data_vector_dic_high, noise_vector_dic_high = get_nx2pt_data_vectors_and_cov(experiment, zmin, zmax, zbinwidth, ell = ell, cosmo_param_dict = param_dict_high, include_cmb_lensing = include_cmb_lensing, zbin_mid_arr = zbin_mid_arr_high, dndz_lensgalsforclus_dic = dndz_lensgalsforclus_dic_high, dndz_sourcegalsforshear_dic = dndz_sourcegalsforshear_dic_high)

        derivative_vector_dic[ppp] = {}
        for obskey in data_vector_dic_low:
            derivative_vector_dic[ppp][obskey] = {}
            for z1z2 in data_vector_dic_low[obskey]:
                curr_data_vector_low = data_vector_dic_low[obskey][z1z2]
                curr_data_vector_high = data_vector_dic_high[obskey][z1z2]
                ##print(curr_data_vector_low)
                ##print(curr_data_vector_high)
                deriv_val = ( curr_data_vector_high - curr_data_vector_low ) / (2 * pstepval)
                derivative_vector_dic[ppp][obskey][z1z2] = deriv_val
                ##print(deriv_val); sys.exit()

    return derivative_vector_dic

def get_nx2t_cov(ell, data_vector_dic_original, zbin_mid_arr, fsky_dic, obs_key_arr = ['cl_s_s', 'cl_g_s', 'cl_g_g'], noise_vector_dic = None):

    import copy

    data_vector_dic = {}
    for obskey in data_vector_dic_original:
        data_vector_dic[obskey] = {}
        for z1z2 in data_vector_dic_original[obskey]:
            z1, z2 = z1z2
            data_vector_dic[obskey][(z1, z2)] = data_vector_dic[obskey][(z2, z1)] = data_vector_dic_original[obskey][(z1, z2)]
    if noise_vector_dic is not None:
        noise_vector_dic_original = copy.deepcopy(noise_vector_dic)
        noise_vector_dic = {}
        for obskey in noise_vector_dic_original:
            noise_vector_dic[obskey] = {}
            for z1z2 in noise_vector_dic_original[obskey]:
                z1, z2 = z1z2
                noise_vector_dic[obskey][(z1, z2)] = noise_vector_dic[obskey][(z2, z1)] = noise_vector_dic_original[obskey][(z1, z2)]

    full_cov_mat_dic = {}

    all_zcombs = [(cntr1, cntr2) for cntr1 in range(len(zbin_mid_arr)) for cntr2 in range(len(zbin_mid_arr)) if cntr1<=cntr2]
    all_combs_ss = ['cl_s_s_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gg = ['cl_g_g_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gs = np.unique( ['cl_g_s_z%s_z%s' %(z1z2[0], z1z2p[1]) for z1z2 in all_zcombs for z1z2p in all_zcombs] ).tolist()
    all_combs = all_combs_ss + all_combs_gg + all_combs_gs
    cov_mat_ndim = len( all_combs )
    ##print(len(all_combs_ss), len(all_combs_gg), len(all_combs_gs), cov_mat_ndim); sys.exit()
    ##print(cov_mat_ndim)
    if 'cl_k_k' in data_vector_dic:
        all_combs_ks = ['cl_k_s_z0_z%s' %(z) for z in zbin_mid_arr]
        all_combs_kg = ['cl_k_g_z0_z%s' %(z) for z in zbin_mid_arr]
        all_combs_kk = ['cl_k_k_z0_z0'] #this for kk
        all_combs_k = all_combs_ks + all_combs_kg + all_combs_kk
        all_combs_final = all_combs + all_combs_k
    else:
        all_combs_final = all_combs
    cov_mat_ndim = len(all_combs_final)
    ##print(cov_mat_ndim); sys.exit()

    for elcntr, elval in enumerate( ell ):
        full_cov_mat = np.zeros( (cov_mat_ndim, cov_mat_ndim) )

        for cntr1, xy in enumerate( all_combs_final ):
            for cntr2, wz in enumerate( all_combs_final ):
                obskeyAB, i, j = xy.split('_z')
                obskeyCD, k, l = wz.split('_z')
                i, j, k, l = int(i), int(j), int(k), int(l)
                A, B = obskeyAB.split('_')[1:]
                C, D = obskeyCD.split('_')[1:]
                
                cl_AC = 'cl_%s_%s' %(A, C)
                if cl_AC not in data_vector_dic: cl_AC = 'cl_%s_%s' %(C, A)
                
                cl_BD = 'cl_%s_%s' %(B, D)
                if cl_BD not in data_vector_dic: cl_BD = 'cl_%s_%s' %(D, B)
                
                cl_AD = 'cl_%s_%s' %(A, D)
                if cl_AD not in data_vector_dic: cl_AD = 'cl_%s_%s' %(D, A)
                
                cl_BC = 'cl_%s_%s' %(B, C)
                if cl_BC not in data_vector_dic: cl_BC = 'cl_%s_%s' %(C, B)

                curr_data_val1 = data_vector_dic[cl_AC][(i,k)][elcntr]
                if A == C:
                    curr_data_val1 = curr_data_val1 + noise_vector_dic[cl_AC][(i,k)][elcntr]
                
                curr_data_val2 = data_vector_dic[cl_BD][(j,l)][elcntr]
                if B == D:
                    curr_data_val2 = curr_data_val2 + noise_vector_dic[cl_BD][(j,l)][elcntr]
                
                curr_data_val3 = data_vector_dic[cl_AD][(i,l)][elcntr]
                if A == D:
                    curr_data_val3 = curr_data_val3 + noise_vector_dic[cl_AD][(i,l)][elcntr]
                
                curr_data_val4 = data_vector_dic[cl_BC][(j,k)][elcntr]
                if B == C:
                    curr_data_val4 = curr_data_val4 + noise_vector_dic[cl_BC][(j,k)][elcntr]

                ##print(obskeyAB, obskeyCD, i, j, k, l); ##sys.exit()
                curr_fskyval = np.sqrt( fsky_dic[obskeyAB][(i, j)] * fsky_dic[obskeyCD][(k, l)] )
                #curr_fskyval = np.sqrt( fsky_dic[obskeyAB] * fsky_dic[obskeyCD] )
                ##print(obskeyAB, obskeyCD, i, j, k, l, curr_fskyval); ##sys.exit()

                full_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / curr_fskyval

        ##sys.exit()

        ###to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot, vmin = 0., vmax = 1e-12); colorbar(); show(); sys.exit()

        full_cov_mat = full_cov_mat * ( 1 / (2 * elval + 1) )
        
        if (0):##elcntr == 18: 
            #corr mat
            import scipy.linalg             
            full_inv_cov_mat = linalg.pinv(full_cov_mat)
            to_plot = np.copy(full_cov_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot); colorbar(); show()

            to_plot = np.copy(full_inv_cov_mat)
            to_plot[to_plot==0.] = None
            imshow(to_plot); colorbar(); show()
            sys.exit()
        full_cov_mat_dic[elcntr] = full_cov_mat

    return full_cov_mat_dic

def corr_from_cov(covmat):
    diags = np.sqrt(np.diag(covmat))
    corrmat = np.zeros_like(covmat)
    for i in range(covmat.shape[0]):
        for j in range(covmat.shape[0]):
            corrmat[i, j] = covmat[i, j] / (diags[i] *  diags[j])
    return corrmat

def get_nx2pt_fisher(els, params, cl_deriv_dic, obs_key_arr, cov_mat_nx2pt_dic):

    import scipy.linalg 
    npar = len(params)
    fisher_mat = np.zeros([npar,npar])

    for lcntr, l in enumerate( els ):

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        curr_cov = cov_mat_nx2pt_dic[lcntr]
        if np.sum( curr_cov ) == 0.: continue
        curr_inv_cov = linalg.pinv(curr_cov)

        for (p,p2, pcnt, pcnt2) in param_combinations:

            curr_deriv_vector1 = []
            curr_deriv_vector2 = []
            for curr_obs_key in obs_key_arr:
                ##print(curr_obs_key)
                for z1z2 in cl_deriv_dic[p][curr_obs_key]:
                    z1,z2 = z1z2
                    curr_deriv_vector1.append( cl_deriv_dic[p][curr_obs_key][z1z2][lcntr] )
                    curr_deriv_vector2.append( cl_deriv_dic[p2][curr_obs_key][z1z2][lcntr] )

            ##print(len(curr_deriv_vector1), len(curr_deriv_vector2), curr_inv_cov.shape); sys.exit()

            curr_val = np.dot(curr_deriv_vector1, np.dot( curr_inv_cov, curr_deriv_vector2 ))

            fisher_mat[pcnt2,pcnt] += curr_val            
            ##print( len(curr_deriv_vector1), len(curr_deriv_vector2), curr_inv_cov.shape ); sys.exit()

    return fisher_mat

'''

def get_nx2t_cov(ell, data_vector_dic_original, zbin_mid_arr, fsky_dic, obs_key_arr = ['cl_s_s', 'cl_g_s', 'cl_g_g']):

    import copy

    data_vector_dic = {}
    for obskey in data_vector_dic_original:
        data_vector_dic[obskey] = {}
        for z1z2 in data_vector_dic_original[obskey]:
            z1, z2 = z1z2
            data_vector_dic[obskey][(z1, z2)] = data_vector_dic[obskey][(z2, z1)] = data_vector_dic_original[obskey][(z1, z2)]

    full_cov_mat_dic = {}

    all_zcombs = [(cntr1, cntr2) for cntr1 in range(len(zbin_mid_arr)) for cntr2 in range(len(zbin_mid_arr)) if cntr1<=cntr2]
    all_combs_ss = ['cl_s_s_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gg = ['cl_g_g_z%s_z%s' %(z1z2[0], z1z2[1]) for z1z2 in all_zcombs]
    all_combs_gs = np.unique( ['cl_g_s_z%s_z%s' %(z1z2[0], z1z2p[1]) for z1z2 in all_zcombs for z1z2p in all_zcombs] ).tolist()
    all_combs = all_combs_ss + all_combs_gg + all_combs_gs
    cov_mat_ndim = len( all_combs )
    ##print(len(all_combs_ss), len(all_combs_gg), len(all_combs_gs), cov_mat_ndim); sys.exit()
    ##print(cov_mat_ndim)
    if 'cl_k_k' in data_vector_dic:
        all_combs_ks = ['cl_k_s_z0_z%s' %(z) for z in zbin_mid_arr]
        all_combs_kg = ['cl_k_g_z0_z%s' %(z) for z in zbin_mid_arr]
        all_combs_kk = ['cl_k_k_z0_z0'] #this for kk
        all_combs_k = all_combs_ks + all_combs_kg + all_combs_kk
        all_combs_final = all_combs + all_combs_k
        cov_mat_ndim = len(all_combs_final)
    ##print(cov_mat_ndim); sys.exit()

    for elcntr, elval in enumerate( ell ):
        full_cov_mat = np.zeros( (cov_mat_ndim, cov_mat_ndim) )
        ###print(full_cov_mat.shape); sys.exit()

        #Diagonal: Shear x shear block  
        curr_len = len(all_combs_ss)
        curr_cov_mat = np.zeros( (curr_len, curr_len) )
        curr_obs_key = 'cl_s_s'
        for cntr1, xy in enumerate( all_combs_ss ):
            for cntr2, wz in enumerate( all_combs_ss ):
                x, y = xy.split('_z')[1:]
                w, z = wz.split('_z')[1:]
                x, y, w, z = int(x), int(y), int(w), int(z)
                curr_data_val1 = data_vector_dic[curr_obs_key][(x,w)][elcntr]
                curr_data_val2 = data_vector_dic[curr_obs_key][(y,z)][elcntr]
                curr_data_val3 = data_vector_dic[curr_obs_key][(x,z)][elcntr]
                curr_data_val4 = data_vector_dic[curr_obs_key][(y,w)][elcntr]

                curr_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / np.sqrt( fsky_dic[curr_obs_key] * fsky_dic[curr_obs_key] )
                ###print(cntr1, cntr2, curr_cov_mat[cntr1, cntr2])

        s, e = 0, curr_len
        full_cov_mat[s:e, s:e] = curr_cov_mat
        ##to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); ##sys.exit()

        #Diagonal: Clustering x Clustering block  
        curr_len = len(all_combs_gg)
        curr_cov_mat = np.zeros( (curr_len, curr_len) )
        curr_obs_key = 'cl_g_g'
        for cntr1, xy in enumerate( all_combs_gg ):
            for cntr2, wz in enumerate( all_combs_gg ):
                x, y = xy.split('_z')[1:]
                w, z = wz.split('_z')[1:]
                x, y, w, z = int(x), int(y), int(w), int(z)
                curr_data_val1 = data_vector_dic[curr_obs_key][(x,w)][elcntr]
                curr_data_val2 = data_vector_dic[curr_obs_key][(y,z)][elcntr]
                curr_data_val3 = data_vector_dic[curr_obs_key][(x,z)][elcntr]
                curr_data_val4 = data_vector_dic[curr_obs_key][(y,w)][elcntr]

                curr_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / np.sqrt( fsky_dic[curr_obs_key] * fsky_dic[curr_obs_key] )

        s = e
        e = s + curr_len
        full_cov_mat[s:e, s:e] = curr_cov_mat
        ###to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); sys.exit()

        #Diagonal: GGL block
        curr_len = len(all_combs_gs)
        curr_cov_mat = np.zeros( (curr_len, curr_len) )
        curr_obs_key = 'cl_g_s'
        for cntr1, xy in enumerate( all_combs_gs ):
            for cntr2, wz in enumerate( all_combs_gs ):
                x, y = xy.split('_z')[1:]
                w, z = wz.split('_z')[1:]
                x, y, w, z = int(x), int(y), int(w), int(z)
                curr_data_val1 = data_vector_dic['cl_g_g'][(x,w)][elcntr]
                curr_data_val2 = data_vector_dic['cl_s_s'][(y,z)][elcntr]
                curr_data_val3 = data_vector_dic[curr_obs_key][(x,z)][elcntr]
                curr_data_val4 = data_vector_dic[curr_obs_key][(y,w)][elcntr]

                curr_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / np.sqrt( fsky_dic[curr_obs_key] * fsky_dic[curr_obs_key] )

        s = e
        e = s + curr_len
        full_cov_mat[s:e, s:e] = curr_cov_mat
        
        ###to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); sys.exit()

        #Off diagonal: Shear x Clustering block
        curr_len1, curr_len2 = len(all_combs_ss), len(all_combs_gg)
        curr_cov_mat = np.zeros( (curr_len1, curr_len2) )
        for cntr1, xy in enumerate( all_combs_ss ):
            for cntr2, wz in enumerate( all_combs_gg ):
                x, y = xy.split('_z')[1:]
                w, z = wz.split('_z')[1:]
                x, y, w, z = int(x), int(y), int(w), int(z)
                curr_data_val1 = data_vector_dic['cl_g_s'][(x,w)][elcntr]
                curr_data_val2 = data_vector_dic['cl_g_s'][(y,z)][elcntr]
                curr_data_val3 = data_vector_dic['cl_g_s'][(x,z)][elcntr]
                curr_data_val4 = data_vector_dic['cl_g_s'][(y,w)][elcntr]

                curr_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / np.sqrt( fsky_dic['cl_g_g'] * fsky_dic['cl_s_s'] )

        s = curr_len1
        e = s + curr_len2
        full_cov_mat[s:e, 0:curr_len1] = curr_cov_mat
        full_cov_mat[0:curr_len1, s:e] = curr_cov_mat
        ###to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); sys.exit()
        
        #Off diagonal: Shear x GGL block
        curr_len1, curr_len2 = len(all_combs_ss), len(all_combs_gs)
        curr_cov_mat = np.zeros( (curr_len1, curr_len2) )
        for cntr1, xy in enumerate( all_combs_ss ):
            for cntr2, wz in enumerate( all_combs_gs ):
                x, y = xy.split('_z')[1:]
                w, z = wz.split('_z')[1:]
                x, y, w, z = int(x), int(y), int(w), int(z)
                curr_data_val1 = data_vector_dic['cl_g_s'][(x,w)][elcntr]
                curr_data_val2 = data_vector_dic['cl_s_s'][(y,z)][elcntr]
                curr_data_val3 = data_vector_dic['cl_s_s'][(x,z)][elcntr]
                curr_data_val4 = data_vector_dic['cl_g_s'][(y,w)][elcntr]

                curr_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / np.sqrt( fsky_dic['cl_s_s'] * fsky_dic['cl_g_s'] )

        s = e
        e = s + curr_len2

        full_cov_mat[s:e, 0:curr_len1] = curr_cov_mat.T
        full_cov_mat[0:curr_len1, s:e] = curr_cov_mat
        ###to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); sys.exit()

        #Off diagonal: Clustering x GGL block
        curr_len1, curr_len2 = len(all_combs_gg), len(all_combs_gs)
        curr_cov_mat = np.zeros( (curr_len1, curr_len2) )
        for cntr1, xy in enumerate( all_combs_ss ):
            for cntr2, wz in enumerate( all_combs_gs ):
                x, y = xy.split('_z')[1:]
                w, z = wz.split('_z')[1:]
                x, y, w, z = int(x), int(y), int(w), int(z)
                curr_data_val1 = data_vector_dic['cl_g_s'][(x,w)][elcntr]
                curr_data_val2 = data_vector_dic['cl_g_g'][(y,z)][elcntr]
                curr_data_val3 = data_vector_dic['cl_g_g'][(x,z)][elcntr]
                curr_data_val4 = data_vector_dic['cl_g_s'][(y,w)][elcntr]

                curr_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4) / np.sqrt( fsky_dic['cl_s_s'] * fsky_dic['cl_g_s'] )

        full_cov_mat[2*curr_len1: 2*curr_len1+curr_len2, curr_len1: 2*curr_len1] = curr_cov_mat.T
        full_cov_mat[curr_len1: 2*curr_len1, 2*curr_len1: 2*curr_len1+curr_len2] = curr_cov_mat
        ###to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); sys.exit()


        if (1):
            curr_len = len(all_combs_final)
            full_cov_mat = np.zeros( (curr_len, curr_len) )
            for cntr1, xy in enumerate( all_combs_final ):
                for cntr2, wz in enumerate( all_combs_final ):
                    obskeyAB, i, j = xy.split('_z')
                    obskeyCD, k, l = wz.split('_z')
                    i, j, k, l = int(i), int(j), int(k), int(l)
                    A, B = obskeyAB.split('_')[1:]
                    C, D = obskeyCD.split('_')[1:]
                    
                    cl_AC = 'cl_%s_%s' %(A, C)
                    if cl_AC not in data_vector_dic: cl_AC = 'cl_%s_%s' %(C, A)
                    
                    cl_BD = 'cl_%s_%s' %(B, D)
                    if cl_BD not in data_vector_dic: cl_BD = 'cl_%s_%s' %(D, B)
                    
                    cl_AD = 'cl_%s_%s' %(A, D)
                    if cl_AD not in data_vector_dic: cl_AD = 'cl_%s_%s' %(D, A)
                    
                    cl_BC = 'cl_%s_%s' %(B, C)
                    if cl_BC not in data_vector_dic: cl_BC = 'cl_%s_%s' %(C, B)

                    curr_data_val1 = data_vector_dic[cl_AC][(i,k)][elcntr]
                    curr_data_val2 = data_vector_dic[cl_BD][(j,l)][elcntr]
                    curr_data_val3 = data_vector_dic[cl_AD][(i,l)][elcntr]
                    curr_data_val4 = data_vector_dic[cl_BC][(j,k)][elcntr]

                    full_cov_mat[cntr1, cntr2] = (curr_data_val1 * curr_data_val2 + curr_data_val3 * curr_data_val4)## / np.sqrt( fsky_dic[curr_obs_key] * fsky_dic[curr_obs_key] )

            to_plot = np.copy(full_cov_mat); to_plot[to_plot==0.] = None; imshow(to_plot); colorbar(); show(); sys.exit()


        full_cov_mat = full_cov_mat * ( 1 / (2 * elval + 1) )
        if (0):##elcntr == 18: 
            #corr mat
            import scipy.linalg             
            full_inv_cov_mat = linalg.pinv(full_cov_mat)
            to_plot = np.copy(full_cov_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot); colorbar(); show()

            to_plot = np.copy(full_inv_cov_mat)
            to_plot[to_plot==0.] = None
            imshow(to_plot); colorbar(); show()

            """
            full_corr_mat = corr_from_cov(full_cov_mat)
            print(full_corr_mat); sys.exit()
            to_plot = np.copy(full_corr_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot, vmin = -1, vmax = 1.); colorbar(); show()
            """
            sys.exit()
        full_cov_mat_dic[elcntr] = full_cov_mat

    return full_cov_mat_dic

def get_nx2t_cov_work(ell, data_vector_dic, zbin_mid_arr, fsky_dic):

    full_cov_mat_dic = {}
    total_zbins = len( zbin_mid_arr )
    total_spectra = int( total_zbins * (total_zbins+1)/2 )
    total_ellbins = len(ell)
    if 'cl_k_k' in data_vector_dic:
        total_observables = 4
        cov_mat_ndim = (total_observables-1) * total_spectra
        cov_mat_ndim = cov_mat_ndim + 1
    else:
        total_observables = 3
        cov_mat_ndim = total_observables * total_spectra
    k_k_zkey = (1100., 1100.)
    obs_key_arr = ['cl_s_s', 'cl_g_s', 'cl_g_g']
    #obs_key_arr = ['cl_s_s', 'cl_g_g']
    #obs_key_arr = ['cl_g_s']

    for elcntr, elval in enumerate( ell ):
        full_cov_mat = np.zeros( (cov_mat_ndim, cov_mat_ndim) )
        ###print(full_cov_mat.shape); sys.exit()

        if 'cl_k_k' in data_vector_dic: #CMB-lensing
            curr_k_k = data_vector_dic['cl_k_k'][k_k_zkey][elcntr]
            full_cov_mat[-1, -1] = curr_k_k**2./fsky_dic['cl_k_k']
            
            #loop over all zbins and fill the covmat now
            tmpcovindex = 0
            for offdiagiter in range( tot_off_diag_iter ):
                curr_obs_key_split = obs_key_arr[offdiagiter].split('_')
                curr_obs_key = 'cl_k_%s' %(curr_obs_key_split[-1])
                for zcntr, z in enumerate( zbin_mid_arr ):
                    print('What is z1, z2 here? Check'); sys.exit()
                    curr_k_x = data_vector_dic[curr_obs_key][(z1,z2)][elcntr]
                    
                    full_cov_mat[-1, tmpcovindex] = curr_k_x * curr_k_k / fsky_dic[curr_obs_key]
                    full_cov_mat[tmpcovindex, -1] = curr_k_x * curr_k_k / fsky_dic[curr_obs_key]
                    tmpcovindex += 1
        
        #off-diagonals
        all_combs = [(cntr1, cntr2) for cntr1 in range(len(zbin_mid_arr)) for cntr2 in range(len(zbin_mid_arr)) if cntr1<=cntr2]
        tot_off_diag_iter = len(obs_key_arr)
        for offdiagiter1 in range( tot_off_diag_iter ):
            for offdiagiter2 in range( tot_off_diag_iter ):
                ##if offdiagiter1 == offdiagiter2: continue
                obs_key1, obs_key2 = obs_key_arr[offdiagiter1], obs_key_arr[offdiagiter2]
                curr_cov_mat = np.zeros( (total_spectra, total_spectra) )
                for cntr1, z1z2 in enumerate( all_combs ):
                    for cntr2, z1z2p in enumerate( all_combs ):
                        z1, z2 = z1z2
                        z1p, z2p = z1z2p
                        """
                        if obs_key1 == 'cl_g_s':
                            curr_g_s = data_vector_dic[obs_key1][(z1,z2)][elcntr]
                            curr_g_g = data_vector_dic['cl_g_g'][(z1,z2)][elcntr]
                            curr_s_s = data_vector_dic['cl_s_s'][(z1,z2)][elcntr]
                            curr_data_val1 = 0.5 * ( curr_g_s**2. + curr_g_g * curr_s_s )
                            if curr_g_s == 0.: #bin excluded in the final likelihoods
                                curr_data_val1 = 0
                        else:
                            curr_data_val1 = data_vector_dic[obs_key1][(z1,z2)][elcntr]
                        if obs_key2 == 'cl_g_s':
                            curr_g_s = data_vector_dic[obs_key2][(z1p,z2p)][elcntr]
                            curr_g_g = data_vector_dic['cl_g_g'][(z1p,z2p)][elcntr]
                            curr_s_s = data_vector_dic['cl_s_s'][(z1p,z2p)][elcntr]
                            curr_data_val2 = 0.5 * ( curr_g_s**2. + curr_g_g * curr_s_s )
                            if curr_g_s == 0.: #bin excluded in the final likelihoods
                                curr_data_val2 = 0
                        else:
                            curr_data_val2 = data_vector_dic[obs_key2][(z1p,z2p)][elcntr]
                        """

                        if obs_key1 == obs_key2 and obs_key1 == 'cl_g_s':
                            curr_g_s = data_vector_dic[obs_key1][(z1,z2p)][elcntr]
                            curr_g_g = data_vector_dic['cl_g_g'][(z1,z2p)][elcntr]
                            curr_s_s = data_vector_dic['cl_s_s'][(z1,z2p)][elcntr]
                            curr_data_val = 0.5 * ( curr_g_s**2. + curr_g_g * curr_s_s )
                            if curr_g_s == 0.: #bin excluded in the final likelihoods
                                curr_data_val = 0.
                            curr_cov_mat[(cntr1, cntr2)] = curr_data_val / np.sqrt( fsky_dic[obs_key1] * fsky_dic[obs_key2] )
                        else:
                            curr_data_val1 = data_vector_dic[obs_key1][(z1,z2)][elcntr]
                            curr_data_val2 = data_vector_dic[obs_key2][(z1p,z2p)][elcntr]
                            curr_cov_mat[(cntr1, cntr2)] = curr_data_val1 * curr_data_val2 / np.sqrt( fsky_dic[obs_key1] * fsky_dic[obs_key2] )

                        if (0):##curr_cov_mat[(cntr1, cntr2)] != 0:
                            print(cntr1, cntr2, z1z2, z1z2p, curr_data_val1, curr_data_val2)
                            ###print(data_vector_dic[obs_key1][(z1,z2)][elcntr], data_vector_dic[obs_key2][(z1p,z2p)][elcntr]); sys.exit()
                        if (0):##obs_key1 != obs_key2:
                            print(obs_key1, obs_key2, curr_data_val1, curr_data_val2, z11z21, z21z22)
                    
                ###imshow(curr_cov_mat); colorbar(); title(r'%s x %s' %(obs_key1, obs_key2)); show(); ##sys.exit()
                s1 = total_spectra * offdiagiter1
                e1 = s1 + total_spectra
                s2 = total_spectra * offdiagiter2
                e2 = s2 + total_spectra
                ##print(s1, e1, s2, e2)
                full_cov_mat[s1:e1, s2:e2] = curr_cov_mat

        full_cov_mat = full_cov_mat * ( 1 / (2 * elval + 1) )
        if (0):##elcntr == 18: 
            #corr mat
            import scipy.linalg             
            full_inv_cov_mat = linalg.pinv(full_cov_mat)
            to_plot = np.copy(full_cov_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot); colorbar(); show()

            to_plot = np.copy(full_inv_cov_mat)
            to_plot[to_plot==0.] = None
            imshow(to_plot); colorbar(); show()

            """
            full_corr_mat = corr_from_cov(full_cov_mat)
            print(full_corr_mat); sys.exit()
            to_plot = np.copy(full_corr_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot, vmin = -1, vmax = 1.); colorbar(); show()
            """
            sys.exit()
        full_cov_mat_dic[elcntr] = full_cov_mat

    return full_cov_mat_dic
'''
"""
def get_nx2t_cov_v2(ell, data_vector_dic, zbin_mid_arr, fsky_dic):

    full_cov_mat_dic = {}
    total_zbins = len( zbin_mid_arr )
    total_ellbins = len(ell)
    if 'cl_k_k' in data_vector_dic:
        total_observables = 4
        cov_mat_ndim = (total_observables-1) * total_zbins
        cov_mat_ndim = cov_mat_ndim + 1
    else:
        total_observables = 3
        cov_mat_ndim = total_observables * total_zbins
    k_k_zkey = (1100., 1100.)
    obs_key_arr = ['cl_s_s', 'cl_g_s', 'cl_g_g']

    for elcntr, elval in enumerate( ell ):
        full_cov_mat = np.zeros( (cov_mat_ndim, cov_mat_ndim) )
        ##print(full_cov_mat.shape); sys.exit()

        if 'cl_k_k' in data_vector_dic: #CMB-lensing
            curr_k_k = data_vector_dic['cl_k_k'][k_k_zkey][elcntr]
            full_cov_mat[-1, -1] = curr_k_k**2./fsky_dic['cl_k_k']
            
            #loop over all zbins and fill the covmat now
            tmpcovindex = 0
            for offdiagiter in range( tot_off_diag_iter ):
                curr_obs_key_split = obs_key_arr[offdiagiter].split('_')
                curr_obs_key = 'cl_k_%s' %(curr_obs_key_split[-1])
                for zcntr, z in enumerate( zbin_mid_arr ):
                    curr_k_x = data_vector_dic[curr_obs_key][(z1,z2)][elcntr]
                    
                    full_cov_mat[-1, tmpcovindex] = curr_k_x * curr_k_k / fsky_dic[curr_obs_key]
                    full_cov_mat[tmpcovindex, -1] = curr_k_x * curr_k_k / fsky_dic[curr_obs_key]
                    tmpcovindex += 1
        
        #off-diagonals
        tot_off_diag_iter = len(obs_key_arr)
        for offdiagiter1 in range( tot_off_diag_iter ):
            for offdiagiter2 in range( tot_off_diag_iter ):
                ##if offdiagiter1 == offdiagiter2: continue
                obs_key1, obs_key2 = obs_key_arr[offdiagiter1], obs_key_arr[offdiagiter2]
                curr_cov_mat = np.zeros( (total_zbins, total_zbins) )
                for z1cntr, z1 in enumerate( zbin_mid_arr ):
                    for z2cntr, z2 in enumerate( zbin_mid_arr ):
                        if obs_key1 ==  obs_key2 and obs_key1 == ['cl_g_s']:
                            curr_g_s = data_vector_dic[obs_key1][(z1,z2)][elcntr]
                            curr_g_g = data_vector_dic['cl_g_g'][(z1,z2)][elcntr]
                            curr_s_s = data_vector_dic['cl_s_s'][(z1,z2)][elcntr]
                            curr_cov_mat[(z1cntr, z2cntr)] = 0.5 * ( curr_g_s**2. + curr_g_g * curr_s_s ) / fsky_dic[obs_key1]
                        else:
                            curr_data_val1 = data_vector_dic[obs_key1][(z1,z2)][elcntr]
                            curr_data_val2 = data_vector_dic[obs_key2][(z1,z2)][elcntr]
                            ###print(curr_data_val1, curr_data_val2, curr_data_val1 * curr_data_val2); sys.exit()
                            curr_cov_mat[(z1cntr, z2cntr)] = curr_data_val1 * curr_data_val2 / fsky_dic[obs_key1]
                
                s1 = total_zbins * offdiagiter1
                e1 = s1 + total_zbins
                s2 = total_zbins * offdiagiter2
                e2 = s2 + total_zbins
                full_cov_mat[s1:e1, s2:e2] = curr_cov_mat
                imshow(curr_cov_mat); colorbar();show(); sys.exit()

        full_cov_mat = full_cov_mat * ( 2 / (2 * elval + 1) )
        if (0): 
            #corr mat
            full_corr_mat = corr_from_cov(full_cov_mat)
            to_plot = np.copy(full_cov_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot); colorbar(); show()

            to_plot = np.copy(full_corr_mat)
            to_plot[to_plot==0.] = None
            
            imshow(to_plot, vmin = -1, vmax = 1.); colorbar(); show()
            sys.exit()
        full_cov_mat_dic[elcntr] = full_cov_mat

    return full_cov_mat_dic

def get_ggl_bins_from_binny():
    from binny import NZTomography

    lens = NZTomography()
    lens.build_survey_bins("lsst", role="lens", year="1")

    source = NZTomography()
    source.build_survey_bins("lsst", role="source", year="1")

    spec = {
        "topology": {"name": "pairs_cartesian"},
        "filters": [
            {
                "name": "score_relation",
                "score": "peak",
                "pos_a": 0,
                "pos_b": 1,
                "relation": "gt",
            },
            {
                "name": "overlap_fraction",
                "threshold": 0.1,
                "compare": "le",
            },
        ],
    }

    pairs = lens.bin_combo_filter(spec, other=source)

    len(pairs)
    print(pairs)
    
    return pairs

def get_nx2t_cov_v1(ell, data_vector_dic, zbin_mid_arr):
    total_observables = 3 ##len( data_vector_dic )
    total_zbins = len( zbin_mid_arr )
    total_ellbins = len(ell)
    cov_mat_ndim = (total_observables-1) * total_zbins
    if 'cl_k_k' in data_vector_dic:
        cov_mat_ndim = cov_mat_ndim + 1

    for elcntr, elval in enumerate( ell ):
        full_cov_mat = np.zeros( (cov_mat_ndim, cov_mat_ndim) )

        #diagonals
        for obscntr, obs_key in enumerate( ['cl_g_g', 'cl_s_s'] ):
            curr_cov_mat = np.zeros( (total_zbins, total_zbins) )
            for z1cntr, z1 in enumerate( zbin_mid_arr ):
                for z2cntr, z2 in enumerate( zbin_mid_arr ):
                    curr_data_val = data_vector_dic[obs_key][(z1,z2)][elcntr]
                    curr_cov_mat[(z1cntr, z2cntr)] = curr_data_val**2.
            s = obscntr * total_zbins
            e = s + total_zbins
            full_cov_mat[s:e, s:e] = curr_cov_mat
            ##print(curr_cov_mat)


        #off-diagonals
        for obscntr, obs_key in enumerate( ['cl_g_s'] ):
            curr_cov_mat = np.zeros( (total_zbins, total_zbins) )
            for z1cntr, z1 in enumerate( zbin_mid_arr ):
                for z2cntr, z2 in enumerate( zbin_mid_arr ):
                    curr_data_val = data_vector_dic[obs_key][(z1,z2)][elcntr]
                    curr_cov_mat[(z1cntr, z2cntr)] = curr_data_val**2.
            s = total_zbins
            e = s + total_zbins
            ##print(curr_cov_mat)
            full_cov_mat[s:e, 0:s] = curr_cov_mat
            full_cov_mat[0:s, s:e] = curr_cov_mat

        #CMB-lensing cross
        
        
        #corr mat
        full_corr_mat = tools_nx2pt.corr_from_cov(full_cov_mat)

        
        imshow(full_cov_mat); colorbar(); show()
        sys.exit()    
"""