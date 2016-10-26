import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

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
ac_GaAs = -7.17 #eV
av_GaAs = -1.16 #eV
c11_GaAs = 1221 #GPa
c12_GaAs = 566 #GPa
c44_GaAs = 600 #GPa

'''InAs'''
#Gamma point energy gap values!
Eg_InAs = 0.417 #eV
Varshni_alpha_InAs = 0.000276 #meV/K
Varshni_beta_InAs = 93 #K
VBO_InAs = -0.59 #eV
ac_InAs = -5.08 #eV
av_InAs = -1.00 #eV
c11_InAs = 832.9 #GPa
c12_InAs = 452.6 #GPa
c44_InAs = 395.9 #GPa

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
#calculates alloy lattice constants, default temperature is 300 K
def calculate_alloy_lattice_constant(Ga_mole_fraction_array, temperature=300):
    alloy_lattice_constants = np.zeros_like(Ga_mole_fraction_array)
    #Lattice parameter values from Vergaftman
    ao_InAs = 6.0583 + math.expm1(2.75E-5) * (temperature - 300)
    ao_GaAs = 5.65325 + math.expm1(3.88e-5) * (temperature -300)
    for n in (0, alloy_lattice_constants.size):
        alloy_lattice_constants[n] = linear_interpolation(ao_InAs, ao_GaAs, Ga_mole_fraction_array[n])

    return alloy_lattice_constants
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates alloy lattice constants, default temperature is 300 K
def calculate_alloy_strain(InGaAs_lattice_constant_array, temperature=300):
    alloy_lattice_constants = np.zeros_like(InGaAs_lattice_constant_array)
    #GaAs substrate assumed Lattice parameter values from Vergaftman
    ao_GaAs = 5.65325 + math.expm1(3.88e-5) * (temperature -300)
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

def calculate_valence_band_offset(Ga_mole_fraction_array):
    '''Note: Since there are almost no reports of temperature variations that
    exceed the experimental uncertainties, in all cases we will take the
    valence band offsets to be independent of T - Vergaftman pp 5854

    -VBO values from Vergaftman are referenced in InSb valence band
    '''

    #Create numpy array for data
    Valence_band_offset_values = np.zeros_like(Ga_mole_fraction_array,dtype=np.float32)

    for n in range (0, (Valence_band_offset_values.size)):
        Valence_band_offset_values[n] = bowing_calculation(VBO_InAs, VBO_GaAs, Ga_mole_fraction_array[n], VBO_bowing)

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
    '''Define constants for calculations'''
    #Array for mole Ga fraction for calculations
    Ga_mole_fraction = np.linspace(0,1,num=1001)
    # temperature of crystal during measurement in Kelvin
    Temperature = 300 

    '''plot In(1-x)Ga(x)As band parameters:'''
    #Calculate unstained band gap values:
    Unstrained_band_gap_energy_eV = calculate_alloy_band_gap(Ga_mole_fraction, Temperature)
    Unstrained_band_gap_energy_nm = convert_energy_to_wavelength(Unstrained_band_gap_energy_eV)

    #Calculate unstrained valence band offsets (eV):
    Unstrained_valence_Band_offset = calculate_valence_band_offset(Ga_mole_fraction)
    #Calculate conduction band values (eV):
        #Note: Valence band offset + band gap = conduction band
    Unstrained_conduction_band_offset = Unstrained_valence_Band_offset + Unstrained_band_gap_energy_eV


    #fig1 generates plot of unstrained alloy band gaps in eV
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel('Ga fraction')
    ax1.set_ylabel('Bandgap (eV)')
    plt.plot(Ga_mole_fraction, Unstrained_band_gap_energy_eV)

    #fig2 generates a plot of unstrained alloy band gaps in nm
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_xlabel('Ga fraction')
    ax2.set_ylabel('Bandgap (nm)')
    plt.plot(Ga_mole_fraction, Unstrained_band_gap_energy_nm)
    plt.axis([0,1,850,3500])
    major_ticks = np.arange(850, 3500, 300)
    ax2.set_yticks(major_ticks)

    #fig3 generates a plot of the unstrained In(1-x)Ga(x)As gamma valley band gap
    #Note: Energy values are referenced to InSb valence band minimum (at 0 eV here)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(1,1,1)
    ax3.set_ylabel('Unstrained In(1-x)Ga(x)As band gap (eV)')
    ax3.set_xlabel('Ga fraction')
    UVBE, = plt.plot(Ga_mole_fraction, Unstrained_valence_Band_offset, label="In(1-x)Ga(x)As VB edge")
    UCBE, = plt.plot(Ga_mole_fraction, Unstrained_conduction_band_offset, label="In(1-x)Ga(x)As CB edge")
    Unstrained_band_structure_legend = plt.legend(handles=[UVBE, UCBE], loc=2)
    plt.show()

main()
