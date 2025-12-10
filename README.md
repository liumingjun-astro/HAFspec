# HAFspec
Author: Mingjun Liu (mjliu@bao.ac.cn), Yilong Wang (wangyilong@nao.cas.cn)

Version 6, released on Dec 8, 2025.

## Introduction
HAFspec is a Monte Carlo (MC) simulation code for computing radiative transfer in hot accretion flows. It was initially developed to calculate the spectra of hot accretion flows in our disk-corona models [1, 2], but can also be applied to more generalized cases. In HAFspec, we focus on the radiative transfer along the vertical direction (z-direction) in an axisymmetric accretion flow (r-direction).

window.py: giving input parameter (the structure data of accretion flows should be given in a separate file).

main_cal.py: main function for spectral calculation.

method.py: recording functions describing radiative processes.

const.py: recording physical constants and pre-computed data for scattering.

## Theory
The basic theory of the MC method for spectral calculation has been extensively investigated by [3-7]. Here we give a brief description. In general, five radiative processes in accretion flows have been included in HAFspec, i.e., bremsstrahlung, bremsstrahlung self-absorption, synchrotron radiation, synchrotron self-absorption, and (inverse-) Compton scattering. Since we only focus on unabsorbed photons, the entire process can be separated as follows.

Step 1. calculate spectral emissivity of bremsstrahlung and synchrotron through Eqs.(3.4-3.8) in [8], Eq.(15) in [9] and Eqs.(33-37) in [5].

Step 2. treat self-absorption to give seed photons of scattering following Eq.(26) in [5].

Step 3. calculate Compton scattering via the MC method.

## Algorithm
(1) Seed photon production
The first and second steps give a distribution of seed photons in frequncey-radial space N(nu, r). However, the distribution N(nu, r) exhibits enormous variations in magnitude, which makes direct use of the rejection sampling method highly inefficient. Therefore, we sample seed photons as their spectral energy distribution (SED) and eliminate nu-r grids whose expected number of super-photon (the package of several identical photons) is less than one. Correspondingly, each super-photon is assigned a weight.

(2) Scattering
For each super-photon, it will lose a portion exp(-l/lambad) during each scattering. Once its remaining portion is less than a critical value, we stop the scattering process to treat the next super-photon. All of the escaped part of photons are collected to form the emergent spectrum. In each scattering, the Comptonization is treated following Sec 4.c in [3], Sec 9.5 in [4] and Eqs.(2.20, 2.21, 3.35-3.45) in [7].

The entire process of HAFspec is summarized as follows.

Read data from the input file and window.py ---- main_cal.run_spec() 

|   find the input data file ---- lines 17-96 in main_cal.py & main_cal.find_name()
    
Spectral calculation ---- main_cal.main()

|   check data ---- lines 216-249 in main_cal.py

|   |   obtain frequency ---- main_cal.find_domain(), method.find_nu(), method.cal_nu()

|   |   obtain geometry ---- lines 257-266 in main_cal.py

|   |   seed photons ---- lines 268-273 in main_cal.py & method.seed_photon()

|   |   |   calculate emissivity ---- method.emissivity(), method.gaunt(), method.i_m()

|   |   |   self-absorption ---- lines 32-41 in method.py & method.blackbody()

|   |   assign weight ---- lines 280-281 in main_cal.py & main_cal.get_number()

|   |   scattering ---- lines 283-297 in main_cal.py

|   |   |   sample data ---- method.get_soft()

|   |   |   scattering for each super-photon ---- method.scattering()

|   |   |   |   compute escape probability ---- method.mean_free_path() using pre-computed data by method.get_mfp_data()

|   |   |   |   collect escaped photons ---- lines 329-332 in method.py

|   |   |   |   Comptonization --- method.compton(), method.find_moment(), method.rand_unit_vector()

|   | recording photon information ---- lines 300-319 in main_cal.py
    
Save results ---- lines 104-183 in main_cal.py
    
## Input and output
(1) Set black hole mass and element abundance in window.py

(2) Set frequency-grid and super-photon number in window.py

(3) Set the path of input data in window.py

(4) The path of output data is the same as that of input data.

(5) The input data of accretion flow structure is a (n * 7) dat or csv file, where n is the number of grids on r-direction, 7 rows are radial distence R (cm), ion temperature Ti, electron temperature Te, disk temperature Td, scattering optical depth tau_es, height (cm) and magnetic field (Gs). The disk temperature Td is the effective temperature of the accretion disk in the two-phase accretion model. If you only focus on the hot accretion flow, set each value of Td to zero.

## Reference
[1] Liu M. , Liu B. F., Wang Y., Cheng H., Yuan W., 2025, MNRAS, 539, 69

[2] Wang Y., Liu B. F., Liu M. , 2025, MNRAS, 544, 1833

[3] Pozdniakov L. A. , Sobol I. M., Siuniaev R. A., 1977, Astron. Zh., 54, 1246

[4] Pozdnyako v L. A. , Sobol I. M., Syunyaev R. A., 1983, Astrophys. Space Phys. Res., 2, 189

[5] Manmoto T. , Mineshige S., Kusunose M., 1997, ApJ , 489, 791

[6] Qiao E. , Liu B. F., 2012, ApJ, 744, 145

[7] Ghosh H. , 2013, preprint (arXiv:1307.3635)

[8] Narayan R. , Yi I., 1995b, ApJ , 452, 710

[9] Greene J. , 1959, ApJ , 130, 693
