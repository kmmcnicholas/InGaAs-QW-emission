import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
#Important equations:
#*---*---*---*---*---*---*---*---*---*---*#
'''
Generic eqation for calculating a property with a bowing parameter:
    P(A{1-x}B{x}) =  (1-x)*P(A) + (x)*P(B) - x*(1-x)*C
Where:
    A and B are the binary compoints
    x is the molfraction of component B
    P is the property of the material
    C is the bowing parameter of property P
'''
#*---*---*---*---*---*---*---*---*---*---*#
'''
Equation for temperature dependence (Varshni equation):
    Eg(T) = Eg(T=0) - (alpha*T^2)/(T+beta)
Where:
    Eg(T) is the band gap at the desired temperature
    Eg(T=0) is the band gap at room temperature
    T is the desired temperature
    alpha, beta are fits to experimental data (Varshni parameters, see Vurgaftman)
'''
#*---*---*---*---*---*---*---*---*---*---*#

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
# Define constants:
#*---*---*---*---*---*---*---*---*---*---*#
#Universal constants
#h = 4.135667662*(10**−15) #eV*S
#c = 2.99792458*(10**17) #nm/S
hc = 1239.842 #(ev*nm)

#Binary Parameters (from Vergaftman):
'''GaAs'''
#Gamma point energy gap values!
Eg_GaAs = 1.519 #eV (0 K)
Varshni_alpha_GaAs = 0.0005405 #meV/K
Varshni_beta_GaAs = 204 #K
VBO_GaAs = -0.80 #eV
ao_GaAs

'''InAs'''
#Gamma point energy gap values!
Eg_InAs = 0.417 #eV
Varshni_alpha_InAs = 0.000276 #meV/K
Varshni_beta_InAs = 93 #K
VBO_InAs = -0.59 #eV
ao_InAs

#*---*---*---*---*---*---*---*---*---*---*#
#Ternary bowing parameters (Vergaftman)
'''InGaAs'''
VBO_bowing = -0.38 # eV
Eg_Gamma_bowing = 0.477 #eV
Eg_L_bowing = 0.33 #eV
Eg_X_bowing = 1.4 #eV
Delta_SO_bowing = 0.15 #eV
Ep_bowing = -1.48 #eV
a_c_bowing = 2.61 #eV
meff_hh_100_bowing = -0.145 #m_e
meff_lh_100_bowing = 0.0202 #m_e
F_bowing = 1.77 #dimensionless
luttinger32_bowing = 0.481 #m_e
#*---*---*---*---*---*---*---*---*---*---*#

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
# Define auxiliary funtions

#_________________________________________________________________________#
#Performs linear interpolation between point a (0) and b (1) at point 0<=x<=1 (Vagard's Law)
def linear_interpolation(Property_A, Property_B, x):
    #Make sure to pass values in the correct order!!
    #For In(1-x)Ga(x)As, property A should be In property
    Value = (1-x)*Property_A + (x)*Property_B
    return Value
#_________________________________________________________________________#

#_________________________________________________________________________#
#Performs calculation of property with a bowing parameter
def bowing_calculation(Property_A, Property_B, x, Bowing_param):
    #Make sure to pass values in the correct order!!
    #P(A{1-x}B{x}) =  (1-x)*P(A) + (x)*P(B) - x*(1-x)*C
    #For In(1-x)Ga(x)As, property A should be In property
    Value = (1-x)*Property_A + x*Property_B - x*(1-x)*Bowing_param
    return Value
#_________________________________________________________________________#

#_________________________________________________________________________#
#converts energy valuves in eV to wavelength in nm
def convert_energy_to_wavelength(bandgap_array):
    wavelength = hc / bandgap_array
    return wavelength
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates alloy lattice constants
def calculate_alloy_lattice_constant(Ga_mole_fraction_array, temperature=300):
    alloy_lattice_constants = np.zeros_like(Ga_mole_fraction_array)

    for n in (0, alloy_lattice_constants.size):
        alloy_lattice_constants[n] = linear_interpolation(ao_InAs, ao_GaAs, Ga_mole_fraction_array[n])

    return alloy_lattice_constants
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the band gap over a range of alloy compositions at a user defined temperature (defaults to 300 K)
def calculate_alloy_band_gap(Ga_mole_fraction_array, temperature=300):
    #Create numpy array for data
    Band_gap_values = np.zeros_like(Ga_mole_fraction_array,dtype=np.float32)

    #Calculate binary band gaps at T=temperature
    GaAs_Eg = calculate_temp_dependent_bandgap(Eg_GaAs, Varshni_alpha_GaAs, Varshni_beta_GaAs, temperature)
    InAs_Eg = calculate_temp_dependent_bandgap(Eg_InAs, Varshni_alpha_InAs, Varshni_beta_InAs, temperature)

    for n in range (0, (Band_gap_values.size)):
        Band_gap_values[n] = bowing_calculation(InAs_Eg, GaAs_Eg, Ga_mole_fraction_array[n], Eg_Gamma_bowing)

    print (Band_gap_values)
    return Band_gap_values
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the band gap at a user defined temperature (defaults to 300 K)
def calculate_temp_dependent_bandgap(Bandgap_0K, alpha, beta, temperature=300):
    Bandgap = Bandgap_0K - (alpha*(temperature**2))/(temperature + beta)
    return Bandgap
#_________________________________________________________________________#

def calculate_unstrained_valence_band_offset(Ga_mole_fraction_array):
    #Create numpy array for data
    Valence_band_offset_values = np.zeros_like(Ga_mole_fraction_array,dtype=np.float32)

    for n in range (0, (Valence_band_offset_values.size)):
        Valence_band_offset_values[n] = bowing_calculation(InAs_VBO, GaAs_VBO, Ga_mole_fraction_array[n], VBO_bowing)

    return Valence_band_offset_values
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates confined level
def calculate_confined_level(params):
#TODO - fill in function parameters and return values
    return 0
#_________________________________________________________________________#

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
# Define main funtion

def main():
    #plot InGaAs band parameters:

    #Array for molefraction
    Ga_mole_fraction = np.linspace(0,1,num=1001)

    #Calculate and plot band gap values
    Eg = calculate_alloy_band_gap(Ga_mole_fraction)
    Wavelength = convert_energy_to_wavelength(Eg)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel('Ga fraction')
    ax1.set_ylabel('Bandgap (eV)')
    plt.plot(Ga_mole_fraction, Eg)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_xlabel('Ga fraction')
    ax2.set_ylabel('Bandgap (nm)')
    plt.plot(Ga_mole_fraction, Wavelength)
    plt.axis([0,1,850,3500])
    major_ticks = np.arange(850, 3500, 300)
    ax2.set_yticks(major_ticks)

    plt.show()

main()
