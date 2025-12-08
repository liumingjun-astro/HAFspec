# HAFspec 1D Monte Carlo radiative transfer code for calculating spectra from hot accretion flow
# Version: 6.0 (2025-12-08)
# Author: Mingjun Liu (mjliu@bao.ac.cn), Yilong Wang (wangyilong@nao.cas.cn)
# National Astronomical Observatories, Chinese Academy of Sciences

import main_cal

# Input parameters
m = 1e8  # The mass of accreting star scaled in solar mass
X = 0.75  # Hydrogen mass fraction
Y = 0.25  # Helium mass fraction
Z = [0]  # The mass fraction of mental
A_Z = [0]  # The atomic number of mental

# Control method
nu_min = 1.0e14  # The minimum of photon energy, in unit of Hz
nu_max = 1.0e21  # The maximum of photon energy, in unit of Hz
bins = 80  # The number of bins of frequency
photons = 1000000  # The number of super-photons for Monte Carlo method (~1e4 is suggested)
error = 1e-6  # The error of calculation

# File setting
path = '.../input_file_path/'  # The path of input file
keyword = 'user.dat'  # The common words in the name of input file
fig_format = 'jpg'  # Picture format

if __name__ == "__main__":
    main_cal.run_spec()
