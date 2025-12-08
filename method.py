# Modular: Method

import numpy as np
import math
import random
import scipy.special
from const import *


# The followed part obtains the soft photons for the Comptonization of ADAF and corona
# See Narayan & Yi 1995, ApJ and Manmoto et al. 1997, ApJ for detailed information

def seed_photon(nu_sp, T_e_sp, n_e_sp, B_sp, cof_br_sp, T_d_sp, A_sp, H_sp):
    """Calculate the number of seed photon number

    Args:
        nu_sp: The frequency of photons
        T_e_sp: The temperature of electrons
        n_e_sp: The number density of electron
        B_sp: Magnetic field
        cof_br_sp: Z^2 * n_i / n_e
        T_d_sp: The effective temperature of soft photons for ext-Compton
        A_sp: he one-side surface area of the ring in accretion flow
        H_sp: The half-height of accretion flows

    Return:
        N_self_sp: The number of unscattered seed photons for self-Compton
        N_ext_sp: The number of seed photons for ext-Compton

    """
    # The soft photons for self-Compton, see Manmoto et al. 1997, ApJ, formula [26]
    B_nu_sp = blackbody(nu_sp, T_e_sp)
    if B_nu_sp == 0:
        # Avoid ZeroDivisionError
        F_in_sp = 0
    else:
        chi_nu_sp = emissivity(nu_sp, T_e_sp, n_e_sp, B_sp, cof_br_sp)
        kappa_nu_sp = chi_nu_sp / (4 * math.pi * B_nu_sp)
        tau_abs_sp = (math.pi ** 0.5 / 2) * kappa_nu_sp * H_sp
        F_in_sp = 2 * math.pi / 3 ** 0.5 * B_nu_sp * (1 - math.exp(-2 * 3 ** 0.5 * tau_abs_sp))
    N_self_sp = F_in_sp * A_sp / (h * nu_sp)

    # The soft photons for ext-Compton
    if T_d_sp <= 0:
        N_ext_sp = 0
    else:
        N_ext_sp = math.pi * blackbody(nu_sp, T_d_sp) * A_sp / (h * nu_sp)  # sigma * T_eff^4

    # Avoid not a number
    if np.isnan(N_self_sp) or N_self_sp < 0:
        N_self_sp = 0
    else:
        N_self_sp *= 2  # Two side
    if np.isnan(N_ext_sp) or N_ext_sp < 0:
        N_ext_sp = 0
    else:
        N_ext_sp *= 2  # Two side
    return N_self_sp, N_ext_sp


def emissivity(nu_em, T_e_em, n_e_em, B_em, cof_br_em):
    """Calculate the emissivity of bremsstrahlung and synchrotron radiation

    Args:
        nu_em: The frequency of photons
        T_e_em: The temperature of electrons
        n_e_em: The number density of electron
        B_em: Magnetic field
        cof_br_em: Z^2 * n_i / n_e

    Return:
        The emissivity of bremsstrahlung and synchrotron radiation

    """
    # Bremsstrahlung cooling, see Narayan & Yi 1995, ApJ, formula [3.4]-[3.8]
    theta_e_em = k * T_e_em / (m_e * c ** 2)
    if theta_e_em < 1:
        F_ei = 4 * (2 * theta_e_em / math.pi ** 3) ** 0.5 * (1 + 1.781 * theta_e_em ** 1.34)
        F_ee = 20 / (9 * math.pi ** 0.5) * (44 - 3 * math.pi ** 2) * theta_e_em ** 1.5 * (
                1 + 1.1 * theta_e_em + theta_e_em ** 2 - 1.25 * theta_e_em ** 2.5)
    else:
        F_ei = 9 * theta_e_em / (2 * math.pi) * (math.log(1.123 * theta_e_em + 0.48) + 1.5)
        F_ee = 24 * theta_e_em * (math.log(2 * math.exp(-gamma_E) * theta_e_em) + 1.28)
    q_br_ei_em = cof_br_em * n_e_em ** 2 * sigma_T * c * alpha_f * m_e * c ** 2 * F_ei
    q_br_ee_em = n_e_em ** 2 * c * r_e ** 2 * m_e * c ** 2 * alpha_f * F_ee
    q_br_em = q_br_ei_em + q_br_ee_em
    # see Manmoto et al. 1997, ApJ, formula [33]-[35]
    chi_br_em = q_br_em * gaunt(nu_em, T_e_em) * math.exp(-h * nu_em / (k * T_e_em)) * h / (k * T_e_em)

    # Synchrotron cooling, see Manmoto et al. 1997, ApJ, formula [36] & [37]
    x_em = 4 * math.pi * m_e * c * nu_em / (3 * e * B_em * theta_e_em ** 2)
    bessel_em = scipy.special.kn(2, 1 / theta_e_em)
    if bessel_em <= 10 ** -200:
        chi_syn_em = 0
    else:
        chi_syn_em = 4.43 * 10 ** -30 * 4 * math.pi * n_e_em * nu_em / bessel_em * i_m(x_em)
    if np.isnan(chi_syn_em):
        chi_syn_em = 0
    return chi_br_em + chi_syn_em


def blackbody(nu_bb, T_bb):
    """Blackbody radiation

    Args:
        nu_bb: The frequency of photons
        T_bb: The temperature of electrons

    Return:
        The flux of blackbody radiation

    """
    if T_bb <= 0:
        return 0
    else:
        x_bb = h * nu_bb / k / T_bb
        if x_bb <= 1e-5:
            # The Rayleigh-Jeans Law, avoid ZeroDivisionError, see Rybicki & Lightman 1979, formula [1.53]
            return 2 * h * nu_bb ** 3 / c ** 2 / x_bb
        elif x_bb >= 12:
            # The Wien side Law, avoid math range error, see Rybicki & Lightman 1979, formula [1.54]
            return 2 * h * nu_bb ** 3 / c ** 2 * math.exp(-x_bb)
        else:
            # The Planck Spectrum, see Rybicki & Lightman 1979, formula [1.51]
            return 2 * h * nu_bb ** 3 / c ** 2 / (math.exp(x_bb) - 1)


def i_m(x_im):
    """Function I'(x) in synchrotron, see Narayan & Yi 1995, ApJ, formula [3.12]

    Args:
        x_im: variable

    Return:
        I'(x_im)

    """
    return 4.0505 / x_im ** (1 / 6) * (1 + 0.4 / x_im ** (1 / 4) + 0.5316 / x_im ** (1 / 2)) * math.exp(
        -1.8899 * x_im ** (1 / 3))


def gaunt(nu_ga, T_e_ga):
    """Calculate the Gaunt factor

    Args:
        nu_ga: Frequency of photons
        T_e_ga: Electron temperature

    Return:
        The Gaunt factor

    Ref:
        Greene 1959, ApJ, formula [15]

    """
    x_ga = h * nu_ga / (k * T_e_ga)
    if x_ga <= 1000:
        return 3 ** 0.5 / math.pi * math.exp(x_ga / 2) * scipy.special.kn(0, x_ga / 2)
    else:
        return (3 / math.pi / x_ga) ** 0.5


# The followed part obtains the self-absorption frequency of bremsstrahlung and synchrotron radiation
# See Rybicki & Lightman 1979 and Narayan & Yi 1995, ApJ for detailed information


def cal_nu(nu_cn, T_e_cn, n_e_cn, H_cn, B_cn, R_cn, cof_br_cn, task_cn):
    """The error for compute self-absorption frequency

    Args:
        nu_cn: Frequency of photons
        T_e_cn: Electron temperature
        n_e_cn: Electron number density
        H_cn: The scale height of accretion flow
        B_cn: Magnetic field
        R_cn: Radius
        cof_br_cn: Z^2 * n_i / n_e
        task_cn: 'nu_ff', 'nu_syn'

    Return:
        The error for compute self-absorption frequency

    """
    x_cn = h * nu_cn / k * T_e_cn
    theta_e_cn = k * T_e_cn / (m_e * c ** 2)
    if task_cn == 'nu_ff':
        # Absorption optical depth for free-free process equals to 1, see Rybicki & Lightman 1979, formula [5.18]
        alpha_cof_cn = 4 * e ** 6 / (3 * m_e * h * c) * (
                    2 * math.pi / 3 / k / m_e) ** 0.5 * T_e_cn ** -0.5 * cof_br_cn * n_e_cn ** 2
        stat_cn = 1 - math.exp(-x_cn)
        if stat_cn == 0:
            stat_cn = x_cn
        tau_nu_cn = alpha_cof_cn * nu_cn ** -3 * stat_cn * gaunt(nu_cn, T_e_cn) * H_cn
        return tau_nu_cn - 1

    elif task_cn == 'nu_syn':
        # The critical frequency of synchrotron, see Narayan & Yi 1995, ApJ, formula [3.13]-[3.15]
        nu_0 = e * B_cn / (2 * math.pi * m_e * c)
        x_M = nu_cn / (1.5 * nu_0 * theta_e_cn ** 2)
        bessel_cn = scipy.special.kn(2, 1 / theta_e_cn)
        if bessel_cn == 0:
            return 0
        else:
            return 2.49 * 10 ** -10 * 4 * math.pi * n_e_cn * R_cn / B_cn / theta_e_cn ** 3 / bessel_cn * i_m(x_M) - 1
    else:
        return 0


def find_nu(T_e_fn, n_e_fn, H_fn, B_fn, R_fn, cof_br_fn, er_inf=1e-12, er_x=1e-7, cal_max_d=100):
    """Compute the self-absorption frequency through dichotomy

    Args:
        T_e_fn: Electron temperature
        n_e_fn: Electron number density
        H_fn: The scale height of accretion flow
        B_fn: Magnetic field
        R_fn: Radius
        cof_br_fn: Z^2 * n_i / n_e
        er_inf: The numerical value of infinitesimal quantity
        er_x: The numerical accuracy of characteristic frequency (1e-7 ~ 1e-12 for 16-bit float)
        cal_max_d: The maximum iterations of dichotomy (>=50 for 16-bit float)

    Return:
        The self-absorption frequency

    """
    nu_br_fn, nu_syn_fn = k * T_e_fn / h, k * T_e_fn / h
    for task_fn in ['nu_ff', 'nu_syn']:
        nu_l_fn, nu_r_fn = er_inf, k * T_e_fn / h
        er_l_fn = cal_nu(nu_l_fn, T_e_fn, n_e_fn, H_fn, B_fn, R_fn, cof_br_fn, task_fn)
        er_r_fn = cal_nu(nu_r_fn, T_e_fn, n_e_fn, H_fn, B_fn, R_fn, cof_br_fn, task_fn)
        nu_c_fn = er_inf
        if er_l_fn <= 0:
            nu_c_fn = nu_l_fn
        elif er_r_fn >= 0:
            nu_c_fn = nu_r_fn
        elif er_l_fn * er_r_fn < 0:
            ac_fn = er_x + 1
            count_fn = 0
            while ac_fn > er_x:
                count_fn += 1
                nu_c_fn = (nu_l_fn * nu_r_fn) ** 0.5
                er_c_fn = cal_nu(nu_c_fn, T_e_fn, n_e_fn, H_fn, B_fn, R_fn, cof_br_fn, task_fn)
                ac_fn = math.fabs(1 - nu_l_fn / nu_r_fn)
                if er_c_fn == 0:
                    ac_fn = 0
                elif er_c_fn * er_l_fn > 0:
                    nu_l_fn = nu_c_fn
                    er_l_fn = er_c_fn
                else:
                    nu_r_fn = nu_c_fn
                if count_fn > cal_max_d:
                    ac_fn = 0
        if task_fn == 'nu_ff':
            nu_br_fn = nu_c_fn
        elif task_fn == 'nu_syn':
            nu_syn_fn = nu_c_fn
    return np.maximum(nu_br_fn, nu_syn_fn)


# The followed part uses the Monte Carlo method to calculate the Comptonization of ADAF and corona
# See Pozdnyakov et al. 1977, SvA; Pozdnyakov et al. 1983, ASPRv; Ghosh, 2013, PhD thesis for detailed information

def get_soft(inform_gs, num_file_gs):
    """Get the index of super-photons

    Args:
        inform_gs: The index of nonzero component in num_file_gs
        num_file_gs: The profile of super-photon number

    Return:
        loc_r_gs: The index in radius of super-photons
        loc_nu_gs: The index in frequency of super-photons
        flag_gs: 0 for self-Compton, 1 for ext-Compton

    """
    range_gs = inform_gs.shape[0]  # The number of nonzero grids in the profile of super-photon number
    num_max_gs = np.max(num_file_gs)
    find_gs = False
    count_gs = 0  # Count the times of loop
    cal_max_gs = range_gs * 50  # The maximum times of loop
    loc_list_gs = np.where(num_file_gs == num_max_gs)
    loc_r_gs, loc_nu_gs, flag_gs = loc_list_gs[0][0], loc_list_gs[1][0], loc_list_gs[2][0]
    while not find_gs:
        # Reject method
        loc_gs = random.randint(0, range_gs - 1)
        loc_r_gs, loc_nu_gs, flag_gs = inform_gs[loc_gs, 0], inform_gs[loc_gs, 1], inform_gs[loc_gs, 2]
        accept_gs = num_file_gs[loc_r_gs, loc_nu_gs, flag_gs]
        reject_gs = random.random() * num_max_gs
        if accept_gs > reject_gs:
            find_gs = True
        count_gs += 1
        if count_gs >= cal_max_gs:
            find_gs = True
    return loc_r_gs, loc_nu_gs, flag_gs


def scattering(nu_sca, n_e_sca, T_sca, H_sca, nu_list_sca, w_min_sca=1e-6, complex_sca=False):
    """Calculate the spectrum of one super-photon

    Args:
        nu_sca: The initial frequency of super-photons
        n_e_sca: The number density of electrons
        T_sca: The temperature of electrons
        H_sca: The scale height of accretion flows
        nu_list_sca: The list recording frequency
        w_min_sca: The minimum weight for terminating scattering
        complex_sca: Describe the position of particles

    Return:
        The spectrum of photon number produced by one super-photon

    Ref:
        Pozdnyakov et al. 1977, SvA, Section 6
        Pozdnyakov et al. 1983, ASPRv, Section 9.3 & 9.6

    """
    w0_sca = 1
    omega_sca = rand_unit_vector()  # The initial direction of wavevector for super-photons
    N_nu_sca = np.zeros_like(nu_list_sca)  # Recording the spectrum of one super-photon
    bin_width_sca = math.log10(np.max(nu_list_sca) / np.min(nu_list_sca[0])) / (len(nu_list_sca) - 1)
    if not complex_sca:
        while w0_sca > w_min_sca:
            # Compute the escape probability
            lambda_bar_sca = mean_free_path(nu_sca, T_sca, n_e_sca)  # Obtain the mean free path
            escape_sca = math.exp(-H_sca / lambda_bar_sca)

            # Compute the escape spectrum and the new weight of super-photons
            pos_sca = round(math.log10(nu_sca / nu_list_sca[0]) / bin_width_sca)
            if 0 <= pos_sca < len(nu_list_sca):
                N_nu_sca[pos_sca] += w0_sca * escape_sca
            w0_sca *= 1 - escape_sca

            # Compute the frequency and direction of super-photons after Compton scattering
            p_sca = find_moment(T_sca)
            nu_sca, omega_sca = compton(nu_sca, omega_sca, p_sca)
        return N_nu_sca
    else:
        if omega_sca[2] < 0:
            omega_sca *= -1
        h0_sca = 0  # The initial location of photons
        while w0_sca > w_min_sca:
            # Compute the escape probability
            lambda_bar_sca = mean_free_path(nu_sca, T_sca, n_e_sca)  # Obtain the mean free path
            l_sca = (H_sca - np.sign(omega_sca[2]) * math.fabs(h0_sca)) / math.fabs(omega_sca[2])
            escape_sca = math.exp(-l_sca / lambda_bar_sca)

            # Compute the escape spectrum and the new weight of super-photons
            pos_sca = round(math.log10(nu_sca / nu_list_sca[0]) / bin_width_sca)
            if 0 <= pos_sca < len(nu_list_sca):
                N_nu_sca[pos_sca] += w0_sca * escape_sca
            w0_sca *= 1 - escape_sca

            # Select the random free path and find the coordinate of scattering
            lambda_sca = -lambda_bar_sca * math.log(1 - random.random() * (1 - math.exp(-l_sca / lambda_bar_sca)))
            h0_sca = h0_sca + lambda_sca * omega_sca[2]
            h0_sca = h0_sca % H_sca

            # Compute the frequency and direction of super-photons after Compton scattering
            p_sca = find_moment(T_sca)
            nu_sca, omega_sca = compton(nu_sca, omega_sca, p_sca)
        return N_nu_sca


def compton(nu_comp, omega_comp, p_comp):
    """Calculate the frequency and direction of photon after Compton scattering

    Args:
        nu_comp: The frequency of photon before Compton scattering
        omega_comp: The direction of wavevector for photon before Compton scattering
        p_comp: The momentum of electron before Compton scattering

    Return:
        nu_p_comp: The frequency of photon after Compton scattering
        omega_p_comp: The direction of wavevector for photon after Compton scattering

    Ref:
        Pozdnyakov et al. 1977, SvA, Section 4.c
        Pozdnyakov et al. 1983, ASPRv, Section 9.5
        Ghosh, 2013, PhD thesis, formula [2.20], [2.21], [3.35]-[3.45]

    """
    beta_comp = p_comp / (p_comp ** 2 + (m_e * c) ** 2) ** 0.5
    gamma_comp = (1 - beta_comp ** 2) ** -0.5
    nu_p_comp = nu_comp
    omega_p_comp = omega_comp
    condition_fm = 0
    while condition_fm <= 0:
        # Compute a possible direction of scattering
        x1_comp = random.random()
        mu_p_comp = (beta_comp + 2 * x1_comp - 1) / (1 + beta_comp * (2 * x1_comp - 1))
        phi_p_comp = 2 * math.pi * random.random()

        # Compute the vector and scattering angle
        v_comp = rand_unit_vector()
        rho_comp = (v_comp[0] ** 2 + v_comp[1] ** 2) ** 0.5
        w_comp = np.array([v_comp[1], -v_comp[0], 0]) / rho_comp
        t_comp = np.array([v_comp[0] * v_comp[2], v_comp[1] * v_comp[2], -rho_comp ** 2]) / rho_comp
        omega_p_comp = mu_p_comp * v_comp + (1 - mu_p_comp ** 2) ** 0.5 * (
                math.cos(phi_p_comp) * w_comp + math.sin(phi_p_comp) * t_comp)
        sca_comp = np.dot(omega_comp, omega_p_comp)
        mu_comp = np.dot(omega_comp, v_comp)

        # Compute the factor of Y and the ratio of x
        x_comp = 2 * h * nu_comp / (m_e * c ** 2) * gamma_comp * (1 - mu_comp * beta_comp)
        x_ra_comp = 1 / (1 + h * nu_comp * (1 - sca_comp) / (gamma_comp * m_e * c ** 2 * (1 - mu_p_comp * beta_comp)))
        x_p_comp = x_comp * x_ra_comp
        Y_comp = x_ra_comp ** -2 * (
                1 / x_ra_comp + x_ra_comp + 4 * (1 / x_comp - 1 / x_p_comp) + 4 * (1 / x_comp - 1 / x_p_comp) ** 2)

        # Produce random number and test the selection condition
        condition_fm = Y_comp - 2 * random.random()
        nu_p_comp = x_p_comp / (2 * gamma_comp * (1 - mu_p_comp * beta_comp)) * m_e * c ** 2 / h
    return nu_p_comp, omega_p_comp


def mean_free_path(nu_mfp, T_mfp, n_e_mfp):
    """Calculate the mean free path

    Args:
        nu_mfp: The frequency of photons
        T_mfp: The temperature of electrons
        n_e_mfp: The number density of electrons

    Return:
        The means free path

    Ref:
        Pozdnyakov et al. 1977, SvA, Section 5
        Pozdnyakov et al. 1983, ASPRv, Section 9.4
        Ghosh, 2013, PhD thesis, formula [3.30]-[3.32]

    """
    # linear interpolation
    loc_t_mfp = int(math.log10(T_mfp))
    if loc_t_mfp > 14:
        loc_t_mfp = 14
    loc_nu_mfp = int(math.log10(nu_mfp))
    if loc_nu_mfp > 29:
        loc_nu_mfp = 29
    dt_mfp = np.minimum(math.log10(T_mfp) - loc_t_mfp, 1)
    dnu_mfp = np.minimum(math.log10(nu_mfp) - loc_nu_mfp, 1)
    lambda_mfp = data_mfp[loc_t_mfp][loc_nu_mfp] * dt_mfp * dnu_mfp
    lambda_mfp += data_mfp[loc_t_mfp][loc_nu_mfp + 1] * dt_mfp * (1 - dnu_mfp)
    lambda_mfp += data_mfp[loc_t_mfp + 1][loc_nu_mfp] * (1 - dt_mfp) * dnu_mfp
    lambda_mfp += data_mfp[loc_t_mfp + 1][loc_nu_mfp + 1] * (1 - dt_mfp) * (1 - dnu_mfp)
    return lambda_mfp / n_e_mfp


def find_moment(T_fm):
    """Select the moment of electrons by rejection technique

    Args:
        T_fm: The temperature of electrons

    Return:
        p_fm: The moment of electrons

    Ref:
        Pozdnyakov et al. 1977, SvA, Section 4.a
        Pozdnyakov et al. 1983, ASPRv, Section 9.4
        Ghosh, 2013, PhD thesis, Section 3.3.2

    """
    theta_fm = k * T_fm / (m_e * c ** 2)
    condition_fm = 0
    p_fm = 0
    if theta_fm <= 0.29:
        while condition_fm <= 0:
            x1_fm = random.random()
            x2_fm = random.random()
            x1_p_fm = -1.5 * math.log(x1_fm)
            p_fm = m_e * c * (theta_fm * x1_p_fm * (2 + theta_fm * x1_p_fm)) ** 0.5
            condition_fm = 0.151 * (1 + theta_fm * x1_p_fm) ** 2 * x1_p_fm * (
                    2 + theta_fm * x1_p_fm) * x1_fm - x2_fm ** 2
    else:
        while condition_fm <= 0:
            x1_fm = random.random()
            x2_fm = random.random()
            x3_fm = random.random()
            x4_fm = random.random()
            eta_p_fm = - theta_fm * math.log(x1_fm * x2_fm * x3_fm)
            eta_pp_fm = - theta_fm * math.log(x1_fm * x2_fm * x3_fm * x4_fm)
            p_fm = m_e * c * eta_p_fm
            condition_fm = eta_pp_fm ** 2 - eta_p_fm ** 2 - 1
    return p_fm


def rand_unit_vector():
    """Produce a random unit vector

    Return:
        A random unit vector

    """
    theta_ruv = math.pi * random.random()
    phi_ruv = 2 * math.pi * random.random()
    v1_ruv = math.sin(theta_ruv) * math.cos(phi_ruv)
    v2_ruv = math.sin(theta_ruv) * math.sin(phi_ruv)
    v3_ruv = math.cos(theta_ruv)
    return np.array([v1_ruv, v2_ruv, v3_ruv])


def get_mfp_data(nu_gmd, T_gmd, n_e_gmd):
    """Calculate the 'data_mfp' for obtaining mean free path

    Args:
        nu_gmd: The frequency of photons
        T_gmd: The temperature of electrons
        n_e_gmd: The number density of electrons

    Return:
        The means free path times the number density of electrons,

    Ref:
        Pozdnyakov et al. 1977, SvA, Section 5
        Pozdnyakov et al. 1983, ASPRv, Section 9.4
        Ghosh, 2013, PhD thesis, formula [3.30]-[3.32]

    """
    n_gmd = k * T_gmd / (m_e * c ** 2)
    H_gmd = 2 * h * nu_gmd / (m_e * c ** 2)
    g_gmd = 0
    phi_gmd = 0
    N1_gmd = 300
    for m_gmd in range(1, N1_gmd + 1):
        gamma_m_gmd = 1 - n_gmd * np.log((m_gmd - 0.5) / N1_gmd)
        g_gmd += gamma_m_gmd * (gamma_m_gmd ** 2 - 1) ** 0.5
        x_up_gmd = H_gmd * (gamma_m_gmd + (gamma_m_gmd ** 2 - 1) ** 0.5)
        x_down_gmd = H_gmd * (gamma_m_gmd - (gamma_m_gmd ** 2 - 1) ** 0.5)
        phi_gmd += phi_cs(x_up_gmd) - phi_cs(x_down_gmd)
    g_gmd = g_gmd / (0.375 * sigma_T * n_e_gmd)
    return g_gmd * H_gmd ** 2 / phi_gmd


def phi_cs(x_cs):
    """Calculated the integral of the approximations of the product of photons energy and cross section

    Args:
        x_cs: The dimensionless photon energy

    Return:
        The integral of the approximations of the product of photons energy and cross section

    Ref:
        Pozdnyakov et al. 1977, SvA, Section 5
        Pozdnyakov et al. 1983, ASPRv, Section 9.4
        Ghosh, 2013, PhD thesis, formula [3.23]

    """
    if x_cs <= 0.5:
        return x_cs ** 2 / 6 + 0.047 * x_cs ** 3 - 0.03 * x_cs ** 4 + x_cs ** 2 / 2 / (1 + x_cs)
    elif x_cs <= 3.5:
        return (1 + x_cs) * np.log(1 + x_cs) - 0.94 * x_cs - 0.00925
    else:
        return (1 + x_cs) * np.log(1 + x_cs) + 0.5 * x_cs - 13.16 * np.log(2 + 0.076 * x_cs) + 9.214
