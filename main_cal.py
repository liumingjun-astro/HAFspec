# Modular: main function

import numpy as np
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from const import *
import window
import method


def run_spec():
    """Prepare the data, run the main function and save results for spectrum calculation"""
    R_list_rs, T_i_list_rs, T_e_list_rs, T_d_list_rs, tau_es_list_rs, H_list_rs, B_list_rs = [], [], [], [], [], [], []
    # Obtain the common words in the name of input file
    try:
        key_word_rs = window.keyword
    except AttributeError:
        key_word_rs = '.dat'
    input_list_rs = find_name(window.path, [key_word_rs])  # Obtain all of the input files

    for file_rs in input_list_rs:
        parent_path_rs = os.path.dirname(file_rs)  # Obtain the parents path of one input file
        same_path_file_rs = find_name(parent_path_rs, [key_word_rs])  # Whether one path has more than one file
        # Create the path for saving data
        if len(same_path_file_rs) == 1:
            path_rs = parent_path_rs + '/results/'
            if os.path.exists(path_rs) == 0:
                os.makedirs(path_rs)
        else:
            count_rs = 1
            path_rs = parent_path_rs + '/results_' + str(count_rs) + '/'
            while os.path.exists(path_rs) == 1:
                count_rs += 1
                path_rs = parent_path_rs + '/results_' + str(count_rs) + '/'
            os.makedirs(path_rs)
        name_file_rs = open(path_rs + '/input_info.txt', 'a')
        name_file_rs.write(file_rs)
        name_file_rs.close()

        # Read data from the input file and window.py
        try:
            spec = np.loadtxt(file_rs)
            if len(spec[0, :]) == 7:
                # R, Ti, Te, Td, tau_es, H, B
                run = True
                R_list_rs = spec[:, 0]
                T_i_list_rs = spec[:, 1]
                T_e_list_rs = spec[:, 2]
                T_d_list_rs = spec[:, 3]
                tau_es_list_rs = spec[:, 4]
                H_list_rs = spec[:, 5]
                B_list_rs = spec[:, 6]
            else:
                run = False
                print(file_rs + " may not be the input file")
                print("Please give the data as the format 'R, Ti, Te, Td, tau_es, H, B'.")
        except ValueError:
            run = False
            print(file_rs + " may not be the input file")
        if run:
            try:
                m_rs = window.m
            except AttributeError:
                m_rs = 10
            try:
                X_rs = window.X
            except AttributeError:
                X_rs = 0.75
            try:
                Y_rs = window.Y
            except AttributeError:
                Y_rs = 1 - X_rs
            try:
                Z_rs = window.Z
                A_Z_rs = window.A_Z
            except AttributeError:
                Z_rs, A_Z_rs = [0], [0]
            try:
                nu_min_rs = window.nu_min
                nu_max_rs = window.nu_max
            except AttributeError:
                nu_min_rs, nu_max_rs = 1e14, 1e21
            try:
                bins_rs = window.bins
            except AttributeError:
                bins_rs = 50
            try:
                photons_rs = window.photons
            except AttributeError:
                photons_rs = 10000
            try:
                error_rs = window.error
            except AttributeError:
                error_rs = 1e-6

            # Run calculation
            L_rs, num_rs, flux_rs, en_rs = main(R_list_rs, T_i_list_rs, T_e_list_rs, T_d_list_rs, tau_es_list_rs,
                                                H_list_rs, B_list_rs, m_rs, X_rs, Y_rs, Z_rs, A_Z_rs, nu_min_rs,
                                                nu_max_rs, bins_rs, photons_rs, error_rs)

            # Save results
            head_rs = ['L_Edd', 'Total', 'Self', 'Ext']
            data = pd.DataFrame(L_rs.reshape(1, 4))
            data.to_csv(path_rs + '/total_lum.csv', header=head_rs, index=False)

            head_rs = ['nu(Hz)', 'nu(keV)', 'N(photons/s/Hz)', 'N(self)', 'N(Ext)', 'Seed', 'Seed(self)', 'Seed(Ext)']
            data = pd.DataFrame(num_rs)
            data.to_csv(path_rs + '/number.csv', header=head_rs, index=False)

            head_rs = ['nu(Hz)', 'nu(keV)', 'Lnu(erg/s/Hz)', 'Lnu(self)', 'Lnu(Ext)', 'Seed', 'Seed(self)', 'Seed(Ext)']
            data = pd.DataFrame(flux_rs)
            data.to_csv(path_rs + '/flux.csv', header=head_rs, index=False)

            head_rs = ['nu(Hz)', 'nu(keV)', 'nuLnu/LEdd', 'nuLnu/LEdd(self)', 'nuLnu/LEdd(Ext)', 'Seed', 'Seed(self)',
                       'Seed(Ext)']
            data = pd.DataFrame(en_rs)
            data.to_csv(path_rs + '/nuLnu.csv', header=head_rs, index=False)

            title_rs = r'$m$ = ' + str(window.m) + r'   $L$ = ' + str('%.2e' % L_rs[1]) + r'$L_\mathrm{Edd}$'

            plt.figure(dpi=200)
            plt.loglog(num_rs[:, 0], num_rs[:, 5], c='k', label='Total soft photons')
            plt.loglog(num_rs[:, 0], num_rs[:, 6], linestyle='--', c='b', label='Br + syn')
            plt.loglog(num_rs[:, 0], num_rs[:, 7], linestyle='-.', c='r', label='External')
            plt.title(title_rs)
            plt.xlabel(r'$\nu\mathrm{(Hz)}$')
            plt.ylabel(r'$N_{\nu}$ (photons s$^{-1}$ Hz$^{-1}$)')
            plt.ylim([np.max(num_rs[:, 5]) * 1e-15, np.max(num_rs[:, 5]) * 3])
            plt.tick_params(axis='both', direction="in", which='both', top='on', right='on')
            plt.legend(frameon=False)
            plt.savefig(path_rs + '/seeds_number.' + window.fig_format)
            plt.close()

            plt.figure(dpi=200)
            plt.loglog(en_rs[:, 0], en_rs[:, 5], c='k', label='Total soft photons')
            plt.loglog(en_rs[:, 0], en_rs[:, 6], linestyle='--', c='b', label='Br + syn')
            plt.loglog(en_rs[:, 0], en_rs[:, 7], linestyle='-.', c='r', label='External')
            plt.title(title_rs)
            plt.xlabel(r'$\nu\mathrm{(Hz)}$')
            plt.ylabel(r'$\nu$' + r'$L_{\nu}/L_\mathrm{Edd}$')
            plt.ylim([np.max(en_rs[:, 5]) * 1e-5, np.max(en_rs[:, 5]) * 3])
            plt.tick_params(axis='both', direction="in", which='both', top='on', right='on')
            plt.legend(frameon=False)
            plt.savefig(path_rs + '/seeds_energy.' + window.fig_format)
            plt.close()

            plt.figure(dpi=200)
            plt.loglog(flux_rs[:, 0], flux_rs[:, 2], c='k')
            plt.title(title_rs)
            plt.xlabel(r'$\nu\mathrm{(Hz)}$')
            plt.ylabel(r'$L_{\nu}$ (erg s$^{-1}$ Hz$^{-1}$)')
            plt.ylim([np.max(flux_rs[:, 5]) * 1e-5, np.max(flux_rs[:, 5]) * 3])
            plt.tick_params(axis='both', direction="in", which='both', top='on', right='on')
            plt.savefig(path_rs + '/flux.' + window.fig_format)
            plt.close()

            plt.figure(dpi=200)
            plt.loglog(en_rs[:, 0], en_rs[:, 2] * L_rs[0], c='k', label='Total')
            plt.loglog(en_rs[:, 0], en_rs[:, 3] * L_rs[0], linestyle='--', c='b', label='Self-Compton')
            plt.loglog(en_rs[:, 0], en_rs[:, 4] * L_rs[0], linestyle='-.', c='r', label='Ext-Compton')
            plt.title(title_rs)
            plt.xlabel(r'$\nu\mathrm{(Hz)}$')
            plt.ylabel(r'$\nu$' + r'$L_{\nu}$ (erg s$^{-1}$)')
            plt.ylim([np.max(en_rs[:, 2]) * 1e-5 * L_rs[0], np.max(en_rs[:, 2]) * 3 * L_rs[0]])
            plt.tick_params(axis='both', direction="in", which='both', top='on', right='on')
            plt.legend(frameon=False)
            plt.savefig(path_rs + '/nuLnu.' + window.fig_format)
            plt.close()

            plt.figure(dpi=200)
            plt.loglog(en_rs[:, 0], en_rs[:, 2], c='k', label='Total')
            plt.loglog(en_rs[:, 0], en_rs[:, 3], linestyle='--', c='b', label='Self-Compton')
            plt.loglog(en_rs[:, 0], en_rs[:, 4], linestyle='-.', c='r', label='Ext-Compton')
            plt.title(title_rs)
            plt.xlabel(r'$\nu\mathrm{(Hz)}$')
            plt.ylabel(r'$\nu$' + r'$L_{\nu}/L_\mathrm{Edd}$')
            plt.ylim([np.max(en_rs[:, 2]) * 1e-5, np.max(en_rs[:, 2]) * 3])
            plt.tick_params(axis='both', direction="in", which='both', top='on', right='on')
            plt.legend(frameon=False)
            plt.savefig(path_rs + '/nuLnu_LEdd.' + window.fig_format)
            plt.close()


def main(R_list, T_i_list, T_e_list, T_d_list, tau_es_list, H_list, B_list, m=10, X=0.75, Y=0.25, Z=np.array([0]),
         A_Z=np.array([0]), nu_min=1e14, nu_max=1e21, bins=50, photons=1000, error=1e-6):
    """The main function of spectrum calculation for ADAF and corona

    Args:
        R_list: A list or array recording the radius in unit cm
        T_i_list: A list or array recording ion temperature
        T_e_list: A list or array recording electron temperature
        T_d_list: A list or array recording the effective temperature of soft photons for Ext-Compton
        tau_es_list: Scattering optical depth
        H_list: A list or array recording the scale height
        B_list: A list or array recording magnetic field
        m: The mass of accreting star scaled in solar mass
        X: Hydrogen mass fraction
        Y: Helium mass fraction
        Z: The mass fraction of mental
        A_Z: The atomic number of mental
        nu_min: The minimum of photon energy, in unit of Hz
        nu_max: The maximum of photon energy, in unit of Hz
        bins: The number of bins of frequency
        photons: The number of super-photons for Monte Carlo method
        error: The error of calculation

    Return:
        en_total: [Eddington luminosity, Total luminosity, Luminosity of self-Compton, Luminosity of Ext-Compton]
        num_data: Photon spectrum [nu (Hz), nu (keV), total, self, Ext, total seed, seed-self, seed-Ext]
        flux_data: Luminosity [nu (Hz), nu (keV), total, self, Ext, total seed, seed-self, seed-Ext]
        en_data: nu * L_nu / L_Edd [nu (Hz), nu (keV), total, self, Ext, total seed, seed-self, seed-Ext]

    """
    M = m * m_solar  # The mass of accreting star in unit g
    M_dot_Edd = 4 * math.pi * G * M / (eta_eff * 0.4 * c)  # Eddington accretion rate
    L_Edd = M_dot_Edd * (eta_eff * c ** 2)  # Eddington luminosity
    R_s = 2 * G * M / c ** 2  # Schwarzschild radius

    # Arrange the data format
    N_r = len(R_list)  # The number of radii
    photons = int(photons)
    R_list = np.array(R_list)
    T_i_list = np.array(T_i_list)
    T_e_list = np.array(T_e_list)
    if len(T_d_list) != N_r:
        T_d_list = np.zeros(N_r)
    else:
        T_d_list = np.array(T_d_list)
    tau_es_list = np.array(tau_es_list)  # Scattering optical depth
    H_list = np.array(H_list)
    B_list = np.array(B_list)
    r_list = R_list / R_s  # Radius scaled by Schwarzschild radius
    n_e_list = tau_es_list / sigma_T / H_list  # Electron number density

    # Get chemical parameters
    abundance = np.zeros(len(m_a))  # Element mass abundance
    atom_number = np.linspace(0, len(m_a) - 1, len(m_a))  # Atomic number
    abundance[1] = X
    abundance[2] = Y
    for ele in range(len(Z)):
        abundance[A_Z[ele]] = Z[ele]
    abundance = abundance / np.sum(abundance)
    mu_i = 1 / np.dot(abundance, 1 / np.array(m_a))  # The effective molecular weight of ions
    mu_e = 1 / np.dot(atom_number, abundance / np.array(m_a))  # The effective molecular weight of electrons
    nie_ratio = mu_e / mu_i  # n_i / n_e
    Z_bar = (np.dot(atom_number ** 2, abundance / np.array(m_a)) * mu_i) ** 0.5  # Average charge
    cof_br = Z_bar ** 2 * nie_ratio  # Z^2 * n_i / n_e

    # Produce the grid of frequency and the arrays to record results
    bins_use, nu_list = find_domain(nu_min, nu_max, bins, T_e_list, T_d_list, n_e_list, H_list, B_list, R_list, cof_br)
    dlg_nu = math.log10(np.max(nu_list) / np.min(nu_list)) / (bins_use - 1)
    seed_self_list, seed_ext_list, seed_total_list = np.zeros(bins_use), np.zeros(bins_use), np.zeros(bins_use)
    num_self_list, num_ext_list, num_total_list = np.zeros(bins_use), np.zeros(bins_use), np.zeros(bins_use)

    photon_file = np.zeros([N_r, bins_use, 2])  # The number of seed photons at each radius and each frequency bin
    energy_file = np.zeros([N_r, bins_use, 2])  # nu * L_nu for seed photons
    A_list = np.zeros(N_r)  # The one-side surface area of the ring in accretion flow
    for loc in range(N_r):
        if loc == 0:
            A_list[loc] = math.pi * R_list[loc] * math.fabs(R_list[loc] - R_list[loc + 1])
        elif loc == N_r - 1:
            A_list[loc] = math.pi * R_list[loc] * math.fabs(R_list[loc - 1] - R_list[loc])
        else:
            A_list[loc] = math.pi * R_list[loc] * math.fabs(R_list[loc - 1] - R_list[loc + 1])
        # Obtain the information of seed photons
        for num_nu in range(bins_use):
            photon_file[loc, num_nu, :] = method.seed_photon(nu_list[num_nu], T_e_list[loc], n_e_list[loc], B_list[loc],
                                                             cof_br, T_d_list[loc], A_list[loc], H_list[loc])
            energy_file[loc, num_nu, :] = photon_file[loc, num_nu, :] * h * nu_list[num_nu] * nu_list[num_nu]
        seed_self_list += photon_file[loc, :, 0]  # Count soft photon number for self-Compton
        seed_ext_list += photon_file[loc, :, 1]  # Count soft photon number for ext-Compton

    for num_nu in range(bins_use):
        # From N_nu to N_nu * dnu
        photon_file[:, num_nu, :] *= nu_list[num_nu] * dlg_nu

    # Calculation the spectrum
    photons_use, weight, num_file = get_number(photons, photon_file, energy_file, peak_min_gn=10, error_gn=0.3)
    inform = np.transpose(np.array(np.nonzero(num_file)))  # Extract the index of nonzero component

    for num in range(photons_use):
        loc_r, loc_nu, flag = method.get_soft(inform, num_file)
        one_spec = method.scattering(nu_list[loc_nu], n_e_list[loc_r], T_e_list[loc_r], H_list[loc_r], nu_list,
                                     error) * weight[loc_r, loc_nu, flag] / nu_list / dlg_nu  # from dN to N_nu
        seed_total_list[loc_nu] += weight[loc_r, loc_nu, flag] / nu_list[
            loc_nu] / dlg_nu  # Count the used seed photon number
        num_total_list += one_spec
        if flag == 0:
            num_self_list += one_spec  # Self-Compton
        else:
            num_ext_list += one_spec  # Ext-Compton
        if num % 2000 == 0:
            process = num / photons_use * 100
            print('Task progress: ' + '%.2f' % process + ' % (super-photons: ' + str(num) + '/' + str(
                photons_use) + ').')

    # Recording results
    num_data, flux_data, en_data = np.zeros([bins_use, 8]), np.zeros([bins_use, 8]), np.zeros([bins_use, 8])
    en_total = np.zeros(4)  # Total luminosity

    num_data[:, 0], num_data[:, 1] = nu_list, nu_list / conv_Hz_keV
    flux_data[:, 0], flux_data[:, 1] = nu_list, nu_list / conv_Hz_keV
    en_data[:, 0], en_data[:, 1] = nu_list, nu_list / conv_Hz_keV

    # Number
    num_data[:, 2], num_data[:, 3], num_data[:, 4] = num_total_list, num_self_list, num_ext_list  # Scattering
    num_data[:, 5], num_data[:, 6], num_data[:, 7] = seed_total_list, seed_self_list, seed_ext_list  # Seed

    for col in range(2, 8):
        flux_data[:, col] = num_data[:, col] * h * nu_list  # Luminosity
        en_data[:, col] = flux_data[:, col] * nu_list / L_Edd  # nu * L_nu / L_Edd

    en_total[0] = L_Edd
    for col in range(3):
        en_total[col + 1] = en_data[:, col + 2].sum() * dlg_nu

    return en_total, num_data, flux_data, en_data


def find_domain(nu_min_fd, nu_max_fd, bins_fd, T_e_list_fd, T_d_list_fd, n_e_list_fd, H_list_fd, B_list_fd, R_list_fd,
                cof_br_fd):
    """Adjust the range of frequency domain and produce the grid of frequency

    Args:
        nu_min_fd: The minimum value of frequency
        nu_max_fd: The maximum value of frequency
        bins_fd: The initial number of bins of frequency
        T_e_list_fd: An array recording electron temperature
        T_d_list_fd: An array recording the effective temperature of soft photons for ext-Compton
        n_e_list_fd: An array recording electron number density
        H_list_fd: An array recording scale height
        B_list_fd: An array recording magnetic field
        R_list_fd: An array recording radius
        cof_br_fd: Z^2 * n_i / n_e

    Return:
        bins_use_fd: The number of bins of frequency after adjustment
        nu_list_fd: An array recording frequency

    """
    # The range of frequency from temperature
    T_e_max, loc_max = np.max(T_e_list_fd), np.argmax(T_e_list_fd)
    T_e_min, loc_min = np.min(T_e_list_fd), np.argmin(T_e_list_fd)
    nu_e_max, nu_e_min = 3 * T_e_max * k / h, 3 * T_e_min * k / h

    T_d_max, T_d_min = np.max(T_d_list_fd), np.min(T_d_list_fd)
    nu_d_max, nu_d_min = T_d_max * k / h, T_d_min * k / h
    if T_d_max == 0:
        nu_d_min = float('inf')  # Without ext-Compton

    # The self-absorption frequency of bremsstrahlung and synchrotron
    nu_abs_max = method.find_nu(T_e_max, n_e_list_fd[loc_max], H_list_fd[loc_max], B_list_fd[loc_max],
                                R_list_fd[loc_max], cof_br_fd)
    nu_abs_min = method.find_nu(T_e_min, n_e_list_fd[loc_min], H_list_fd[loc_min], B_list_fd[loc_min],
                                R_list_fd[loc_min], cof_br_fd)

    max_value_fd = np.max([nu_max_fd, 30 * nu_e_max, 30 * nu_d_max])  # 30 times for safety
    min_value_fd = np.min([nu_min_fd, nu_e_min / 1e3, nu_d_min / 1e3, nu_abs_max / 10, nu_abs_min / 10])  # For safety

    # Adjust the range of frequency
    lg_nu_max_fd, lg_nu_min_fd = int(math.log10(max_value_fd)), int(math.log10(min_value_fd))
    dlg_nu_fd = math.log10(nu_max_fd / nu_min_fd) / bins_fd  # Keep the been width
    bins_use_fd = int((lg_nu_max_fd - lg_nu_min_fd) / dlg_nu_fd)
    print('The reasonable range of frequency: 1e' + str(lg_nu_min_fd) + ' ~ 1e' + str(lg_nu_max_fd) + ' Hz')
    print('The reasonable number of bins: ' + str(bins_use_fd))
    nu_list_fd = np.array(10 ** (np.linspace(start=lg_nu_min_fd, stop=lg_nu_max_fd, num=bins_use_fd, endpoint=True)))

    return bins_use_fd, nu_list_fd


def get_number(num_gn, photon_file_gn, energy_file_gn, peak_min_gn=10, error_gn=0.1):
    """Get the suitable values of super-photon number

    Args:
        num_gn: The given number of super-photons
        photon_file_gn: The nd array recording the number of physical photons
        energy_file_gn: The nd array recording the nu * L_nu of physical photons
        peak_min_gn: The minimum value for the peak super-photons number in photon_file_gn
        error_gn: The maximum fraction of loss super-photons in rounding

    Return:
        num_use_gn: The number of super-photons after adjustment
        weight_gn: The profile of super-photon weight
        num_file_gn: The profile of super-photon number

    """
    # The distribution of super-photons follows the nu * L_nu profile
    pro_file_gn = energy_file_gn / energy_file_gn.sum()  # Normalization
    num_file_gn = np.around(num_gn * pro_file_gn)  # The profile of super-photon number
    peak_gn = np.max(num_file_gn)
    num_use_gn = num_file_gn.sum()  # The number of super-photons
    loss_gn = 1 - num_use_gn / num_gn  # Recording the lost super-photon number
    while peak_gn < peak_min_gn and loss_gn > error_gn:
        num_gn *= 2
        num_file_gn = np.around(num_gn * pro_file_gn)
        peak_gn = np.max(num_file_gn)
        num_use_gn = num_file_gn.sum()
        loss_gn = 1 - num_use_gn / num_gn
    num_use_gn = int(num_use_gn)
    print('The reasonable number of super-photons: ' + str(num_use_gn))

    # Calculate weight, physical number / super-photon number
    photon_nonzero_gn = np.where(num_file_gn == 0, 0, photon_file_gn) # Avoid ZeroDivision
    num_nonzero_gn = np.where(num_file_gn == 0, 1, num_file_gn) # Avoid ZeroDivision
    weight_gn = photon_nonzero_gn / num_nonzero_gn
    weight_gn = np.where(np.isnan(weight_gn), 0, weight_gn)

    return num_use_gn, weight_gn, num_file_gn


def find_name(target_path, key_word):
    """Find the file name

    Args:
        target_path: the path of target files
        key_word: a list recording the peculiar words in the name of target folders
`
    Returns:
        name: The list recording the path of target files

    """
    dir_list = os.listdir(target_path)
    name = []
    for find in range(0, len(dir_list)):
        find_file = 1
        for each_key_word in key_word:
            if each_key_word not in dir_list[find]:
                find_file = 0
        if find_file == 1:
            name.append(os.path.join(target_path, dir_list[find]))
        find += 1
    return name
