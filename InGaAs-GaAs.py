#TODO a lot

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve

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
Equation for band gap temperature dependence (Varshni equation):
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
me = 9.10938356e-31 #kg
hbar = 1.054571800e-34 #J*S
JtoeV = 6.242e18 #eV/J

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
b_GaAs = -2.0 # eV
meff_e_gamma_GaAs = 0.067
gamma_1_GaAs = 6.98 #dimensionless
gamma_2_GaAs = 2.06 #dimensionless
gamma_3_GaAs = 2.93 #dimensionless


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
b_InAs = -1.8 #eV
meff_e_gamma_InAs = 0.026 #m_e
gamma_1_InAs = 20.0 #dimensionless
gamma_2_InAs = 8.5 #dimensionless
gamma_3_InAs = 9.2 #dimensionless

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
meff_e_gamma_bowing = 0.0091 #m_e
meff_hh_100_bowing = -0.145 #m_e
meff_lh_100_bowing = 0.0202 #m_e
F_bowing = 1.77 #dimensionless
luttinger32_bowing = 0.481 #m_e
#*---*---*---*---*---*---*---*---*---*---*#

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
# Define auxiliary funtions

#_________________________________________________________________________#
#Performs linear interpolation between point a (0) and b (1) at point 0<=x<=1 (Vagard's Law)
def linear_interpolation(InAs_property, GaAs_property, Ga_mole_fraction):
    #Make sure to pass values in the correct order!!
    Value = (1-Ga_mole_fraction)*InAs_property + (Ga_mole_fraction)*GaAs_property
    return Value
#_________________________________________________________________________#

#_________________________________________________________________________#
#Performs calculation of property with a bowing parameter
def bowing_calculation(InAs_property, GaAs_proptery, Ga_mole_fraction, Bowing_param):
    #Make sure to pass values in the correct order!!
    Value = (1-Ga_mole_fraction)*InAs_property + (Ga_mole_fraction)*GaAs_proptery - Ga_mole_fraction*(1-Ga_mole_fraction)*Bowing_param
    return Value
#_________________________________________________________________________#

#_________________________________________________________________________#
#converts energy valuves in eV to wavelength in nm
def convert_energy_to_wavelength(Bandgap_array):
    Wavelength = hc / Bandgap_array
    return Wavelength
#_________________________________________________________________________#

#_________________________________________________________________________#
#converts energy valuves in eV to wavelength in nm
def convert_joules_to_eV(Joule_energy_array):
    eV_energy_array = JtoeV * Joule_energy_array
    return eV_energy_array
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates alloy lattice constants, default temperature is 300 K
def calculate_alloy_lattice_constant(Ga_mole_fraction_array, Temperature=300):
    Alloy_lattice_constants = np.zeros_like(Ga_mole_fraction_array)
    #Lattice parameter values from Vergaftman
    ao_InAs = 6.0583 + 2.75E-5 * (Temperature - 300)
    ao_GaAs = 5.65325 + 3.88e-5 * (Temperature -300)
    for n in range (0, Alloy_lattice_constants.size):
        Alloy_lattice_constants[n] = linear_interpolation(ao_InAs, ao_GaAs, Ga_mole_fraction_array[n])

    return Alloy_lattice_constants
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates alloy in-plane strain at a given temperature
def calculate_alloy_inplane_strain(Ga_mole_fraction_array, Temperature=300):
    #First calculate lattice constants of alloys
    Alloy_lattice_constants = calculate_alloy_lattice_constant(Ga_mole_fraction_array, Temperature)

    #Second calculate in-plane strain
    Alloy_strain = np.zeros_like(Alloy_lattice_constants)
    #GaAs substrate assumed Lattice parameter values from Vergaftman
    ao_GaAs = 5.65325 + 3.88e-5 * (Temperature -300)

    for n in range (0, Alloy_strain.size):
        Alloy_strain[n] = (ao_GaAs - Alloy_lattice_constants[n]) / Alloy_lattice_constants[n]

    return Alloy_strain
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates alloy out-of-plane strain at a given temperature
def calculate_alloy_outplane_strain(Ga_mole_fraction_array, Temperature=300):
    #First, calculate the inplane strain for the alloys (necessary for out of plane strain)
    Alloy_inplane_strain = calculate_alloy_inplane_strain(Ga_mole_fraction_array, Temperature)

    #Second, calculate out of plane strain
    Alloy_outplane_strain = np.zeros_like(Alloy_inplane_strain)

    for n in range (0, Alloy_outplane_strain.size):
        #Calculate interpolated alloy C11 and C12 parameters
        Alloy_c11 = linear_interpolation(c11_InAs, c12_GaAs, Ga_mole_fraction_array[n])
        Alloy_c12 = linear_interpolation(c12_InAs, c12_GaAs, Ga_mole_fraction_array[n])

        #Calculate out of plan strain using C11 and C12 values
        Alloy_outplane_strain[n] = -2*(Alloy_c12/Alloy_c11)*Alloy_inplane_strain[n]

    return Alloy_outplane_strain
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the del Ec value for strained band shifts (delta Eg = del Ec + P + Q)
def calculate_alloy_del_Ec(Ga_mole_fraction_array,Temperature=300):
    #del Ec = ac*(Exx + Eyy + Ezz) = ac*(2*E|| + Ezz)
    #note: ac has a bowing parameter (Vergaftman)!

    #first, calculate strains for each alloy composition
    Alloy_inplane_strain = calculate_alloy_inplane_strain(Ga_mole_fraction_array, Temperature)
    Alloy_outplane_strain = calculate_alloy_outplane_strain(Ga_mole_fraction_array, Temperature)

    #Second, calculate del Ec for each alloy composition
    Alloy_del_Ec = np.zeros_like(Ga_mole_fraction_array)
    for n in range (0, Alloy_del_Ec.size):
        #Calculate ac using bowing parameter
        Alloy_ac = bowing_calculation(ac_InAs, ac_GaAs, Ga_mole_fraction_array[n], a_c_bowing)
        #calculate del Ec
        Alloy_del_Ec[n] = Alloy_ac*(2*Alloy_inplane_strain[n] + Alloy_outplane_strain[n])

    return Alloy_del_Ec
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the P value for strained band shifts (delta Eg = del Ec + P + Q)
def calculate_alloy_P(Ga_mole_fraction_array,Temperature=300):
    #P = av*(Exx + Eyy + Ezz) = -av*(2*E|| + Ezz)
    # Note that our sign convention for av is different from many other works found in the literature. (Vergaftman)
    # - The negative usually found in front of av (Chuang/UT Optoelectronics) is included in the constants defined above
    #note: ac has NO bowing parameter (Vergaftman)!

    #first, calculate strains for each alloy composition
    Alloy_inplane_strain = calculate_alloy_inplane_strain(Ga_mole_fraction_array, Temperature)
    Alloy_outplane_strain = calculate_alloy_outplane_strain(Ga_mole_fraction_array, Temperature)

    #Second, calculate P for each alloy composition
    Alloy_P = np.zeros_like(Ga_mole_fraction_array)
    for n in range (0, Alloy_P.size):
        #Calculate av using linear interpolation
        Alloy_av = linear_interpolation(av_InAs, av_GaAs, Ga_mole_fraction_array[n])
        #Calculate P
        Alloy_P[n] = Alloy_av*(2*Alloy_inplane_strain[n] + Alloy_outplane_strain[n])

    return Alloy_P
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the Q value for strained band shifts (delta Eg = del Ec + P + Q)
def calculate_alloy_Q(Ga_mole_fraction_array,Temperature=300):
    #Q = (-b/2)*(Exx + Eyy - 2*Ezz) = (-b/2)*(2*E|| + 2*Ezz)
    #note: Q has NO bowing parameter (Vergaftman)!

    #first, calculate strains for each alloy composition
    Alloy_inplane_strain = calculate_alloy_inplane_strain(Ga_mole_fraction_array, Temperature)
    Alloy_outplane_strain = calculate_alloy_outplane_strain(Ga_mole_fraction_array, Temperature)

    #Second, calculate P for each alloy composition
    Alloy_Q = np.zeros_like(Ga_mole_fraction_array)
    for n in range (0, Alloy_Q.size):
        #Calculate av using linear interpolation
        Alloy_b = linear_interpolation(b_InAs, b_GaAs, Ga_mole_fraction_array[n])
        #Calculate P
        Alloy_Q[n] = (-Alloy_b/2)*(2*Alloy_inplane_strain[n] - 2*Alloy_outplane_strain[n])

    return Alloy_Q
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the electron effective mass for InGaAs alloy compositions
def calculate_alloy_me(Ga_mole_fraction_array,Temperature=300):
    #note: me has a bowing parameter (Vergaftman)!
    #Calculate electron effective mass using bowing parameter
    Alloy_me = np.zeros_like(Ga_mole_fraction_array)
    for n in range (0, Alloy_me.size):
        Alloy_me[n] = bowing_calculation(meff_e_gamma_InAs, meff_e_gamma_GaAs, Ga_mole_fraction_array[n], meff_e_gamma_bowing)

    return Alloy_me
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the heavy hole effective mass for InGaAs alloy compositions
def calculate_alloy_mhh(Ga_mole_fraction_array,Temperature=300):
    #(mo/mhh = (gamma1 - 2*gamma2)) - (Vergaftman)
    #note: mhh has NO bowing parameter (Vergaftman)!

    #First, calculate InAs and GaAs heavy hole effective mass
    mhh_InAs = 1/(gamma_1_InAs - 2*gamma_2_InAs)
    mhh_GaAs = 1/(gamma_1_GaAs - 2*gamma_2_GaAs)

    #Second, calculate alloy heavy hole effective mass
    Alloy_mhh = np.zeros_like(Ga_mole_fraction_array)

    for n in range (0, Alloy_mhh.size):
        Alloy_mhh[n] = linear_interpolation(mhh_InAs, mhh_GaAs, Ga_mole_fraction_array[n])

    return Alloy_mhh
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the light hole effective mass for InGaAs alloy compositions
def calculate_alloy_mlh(Ga_mole_fraction_array,Temperature=300):
    #(mo/mlh = (gamma1 + 2*gamma2)) - Vergaftman
    #note: mlh has NO bowing parameter (Vergaftman)!

    #First, calculate InAs and GaAs light hole effective mass
    mlh_InAs = 1/(gamma_1_InAs + 2*gamma_2_InAs)
    mlh_GaAs = 1/(gamma_1_GaAs + 2*gamma_2_GaAs)

    #Second, calculate alloy light hole effective mass
    Alloy_mlh = np.zeros_like(Ga_mole_fraction_array)

    for n in range (0, Alloy_mlh.size):
        Alloy_mlh[n] = linear_interpolation(mlh_InAs, mlh_GaAs, Ga_mole_fraction_array[n])

    return Alloy_mlh
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the band gap over a range of alloy compositions at a user defined temperature (defaults to 300 K)
def calculate_alloy_band_gap(Ga_mole_fraction_array, Temperature=300):
    #Create numpy array for data
    Band_gap_values = np.zeros_like(Ga_mole_fraction_array,dtype=np.float32)

    #Calculate binary band gaps at T=temperature
    GaAs_Eg = calculate_temp_dependent_bandgap(Eg_GaAs, Varshni_alpha_GaAs, Varshni_beta_GaAs, Temperature)
    InAs_Eg = calculate_temp_dependent_bandgap(Eg_InAs, Varshni_alpha_InAs, Varshni_beta_InAs, Temperature)

    for n in range (0, (Band_gap_values.size)):
        Band_gap_values[n] = bowing_calculation(InAs_Eg, GaAs_Eg, Ga_mole_fraction_array[n], Eg_Gamma_bowing)

    return Band_gap_values
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates the band gap at a user defined temperature (defaults to 300 K)
def calculate_temp_dependent_bandgap(Bandgap_0K, alpha, beta, Temperature=300):
    Bandgap = Bandgap_0K - (alpha*(Temperature**2))/(Temperature + beta)
    return Bandgap
#_________________________________________________________________________#

def calculate_valence_band_offset(Ga_mole_fraction_array):
    '''Note: Since there are almost no reports of temperature variations that
    exceed the experimental uncertainties, in all cases we will take the
    valence band offsets to be independent of T - Vergaftman pp 5854

    -VBO values from Vergaftman are referenced in InSb valence band
    '''

    #Create numpy array for data
    Valence_band_offset_values = np.zeros_like(Ga_mole_fraction_array)

    for n in range (0, (Valence_band_offset_values.size)):
        Valence_band_offset_values[n] = bowing_calculation(VBO_InAs, VBO_GaAs, Ga_mole_fraction_array[n], VBO_bowing)

    return Valence_band_offset_values
#_________________________________________________________________________#

#_________________________________________________________________________#
#calculates confined level using the inifinte square well approximation
def calculate_infinite_confined_levels(Ga_mole_fraction_array, Well_width, Temperature = 300):
    #En = (hbar^2/2*m)*(n*pi/Lz)^2
    # n = 1 for first confined level
    #Calculate effective masses
    #electrons
    Electron_effective_masses = calculate_alloy_me(Ga_mole_fraction_array, Temperature)
    Heavy_hole_effective_masses = calculate_alloy_mhh(Ga_mole_fraction_array, Temperature)

    # Calculate electron confinement energies - These results are in Joules!!
    Electron_infinite_confinement_energies = np.zeros_like(Ga_mole_fraction_array)
    for n in range (0, (Electron_infinite_confinement_energies.size)):
        Electron_infinite_confinement_energies[n] = (hbar**2/(2*me*Electron_effective_masses[n]))*((math.pi**2)/(Well_width*1e-10)**2)

    # Calculae heavy hole confinement Electron_infinite_confinement_energies
    Heavy_hole_infinite_confinement_energies = np.zeros_like(Ga_mole_fraction_array)
    for n in range (0, (Heavy_hole_infinite_confinement_energies.size)):
        Heavy_hole_infinite_confinement_energies[n] = (hbar**2/(2*me*Heavy_hole_effective_masses[n]))*((math.pi**2)/(Well_width*1e-10)**2)

    # Convert to eV
    Electron_infinite_confinement_energies = convert_joules_to_eV(Electron_infinite_confinement_energies)
    Heavy_hole_infinite_confinement_energies = convert_joules_to_eV(Heavy_hole_infinite_confinement_energies)

    return (Electron_infinite_confinement_energies, Heavy_hole_infinite_confinement_energies)
#_________________________________________________________________________#

#_________________________________________________________________________#
#numerically calculates the finite well confined level assuming a GaAs barrier
#For analytical solution to finite quantum well see D.A.B. Miller opto 343 notes pp. 8
def calculate_finite_confined_levels(Ga_mole_fraction_array, Well_width, Temperature = 300):
    #First calculate relevant values for barrier layers (GaAs)
    #Calculate GaAs band gap at measurement temperature
    Barrier_mole_fraction = np.array([1])
    Barrier_Eg = calculate_alloy_band_gap(Barrier_mole_fraction, Measurment_temperature)
    Barrier_valence_band_offset = calculate_valence_band_offset(Barrier_mole_fraction)
    Barrier_conduction_band_offset = Barrier_valence_band_offset + Barrier_Eg
    Barrier_me = calculate_alloy_me(Barrier_mole_fraction)
    Barrier_mehh = calculate_alloy_mhh(Barrier_mole_fraction)

    #calculate band parameters for strained alloy
    Alloy_unstrained_band_gap_energy_eV = calculate_alloy_band_gap(Ga_mole_fraction_array, Temperature)

    #Calculate unstrained valence band energy offset (eV):
    Alloy_unstrained_valence_Band_offset = calculate_valence_band_offset(Ga_mole_fraction)
    #Calculate conduction band offsets (eV):
    Alloy_unstrained_conduction_band_offset = Unstrained_valence_Band_offset + Unstrained_band_gap_energy_eV

    #calculate strain shifts in conduction band
    Alloy_del_Ec = calculate_alloy_del_Ec(Ga_mole_fraction_array, Temperature)
    Alloy_strained_conduction_band_offset = Alloy_unstrained_conduction_band_offset + Alloy_del_Ec

    #Calculat strain shifts to valence band edges
    Alloy_P = calculate_alloy_P(Ga_mole_fraction_array, Temperature)
    Alloy_Q = calculate_alloy_Q(Ga_mole_fraction_array, Temperature)
    Alloy_strained_heavy_hole_band_offset = Alloy_unstrained_valence_Band_offset - Alloy_P - Alloy_Q
    Alloy_strained_light_hole_band_offset = Alloy_unstrained_valence_Band_offset - Alloy_P + Alloy_Q

    #Calculate effective shifts in band edge from quantum confinement effects
    Alloy_infinite_confined_levels = calculate_infinite_confined_levels(Ga_mole_fraction_array, Well_width, Temperature)
    Alloy_conduction_band_infinite_confined_levels = Alloy_infinite_confined_levels[0]
    Alloy_valence_band_infinite_confined_levels = Alloy_infinite_confined_levels[1]

    #Calculate Alloy Effetive masses
    Alloy_electron_effective_masses = calculate_alloy_me(Ga_mole_fraction_array, Temperature)
    Alloy_heavy_hole_effective_masses = calculate_alloy_mhh(Ga_mole_fraction_array, Temperature)

    #calcute conined levels:
    #Conduction band quantum well states
    Alloy_conduction_band_finite_levels = np.zeros_like(Ga_mole_fraction_array)
    for n in range(0, Alloy_conduction_band_finite_levels.size):

        if(Ga_mole_fraction_array[n] == 1):
            Alloy_conduction_band_finite_levels[n] = 0
        else:
            #Create empty touple to pass arguments to fsolve()
            Calculation_constants = ()
            #First calculate alloy parameters for fsolve()

            #Barrier electron effective mass
            Calculation_constants[0] = Barrier_me

            #Quantum well electron effective mass
            Alloy_me = Alloy_electron_effective_masses[n]
            Calculation_constants[1] = Alloy_me

            #Quantum well energy barrier
            Energy_barrier = Barrier_conduction_band_offset - Alloy_strained_conduction_band_offset[n]
            Calculation_constants[2] = Energy_barrier

            #Infinite well ground state energy
            Infinite_E1 = Alloy_conduction_band_finite_levelsonduction_band_infinite_confined_levels[n]
            Calculation_constants[3] = Infinite_E1

            #Solve transcendental equation
            Finite_E0 = Infinite_E1 # This is the initial guess for fsolve
            Alloy_conduction_band_finite_levels[n] = fsolve(finite_well_analytical_function, Finite_E0, Calculation_constants)

    #Repeat process for Valence band confined levels
    Alloy_valence_band_finite_levels = np.zeros_like(Ga_mole_fraction_array)
    for n in range(0, Alloy_valence_band_finite_levels.size):

        if(Ga_mole_fraction_array[n] == 1):
            Alloy_valence_band_finite_levels[n] = 0
        else:
            #Create empty touple to pass arguments to fsolve()
            Calculation_constants = ()
            #First calculate alloy parameters for fsolve()

            #Barrier electron effective mass
            Calculation_constants[0] = Barrier_me

            #Quantum well electron effective mass
            Alloy_me = Alloy_heavy_hole_effective_masses[n]
            Calculation_constants[1] = Alloy_mhh

            #Quantum well energy barrier
            Energy_barrier = math.fabs(Barrier_valence_band_offset - Alloy_strained_heavy_hole_band_offset[n])
            Calculation_constants[2] = Energy_barrier

            #Infinite well ground state energy
            Infinite_E1 = Alloy_valence_band_infinite_confined_levels[n]
            Calculation_constants[3] = Infinite_E1

            #Solve transcendental equation
            Finite_E0 = Infinite_E1 # This is the initial guess for fsolve
            Alloy_valence_band_finite_levels[n] = fsolve(finite_well_analytical_function, Finite_E0, Calculation_constants)


    return(Alloy_conduction_band_finite_levels, Alloy_valence_band_finite_levels)
    '''
    Alloy_Confined_conduction_band_offsets = Strained_conduction_band_offset + Conduction_band_infinite_confined_levels
    #Valence band edge values are negative with respect to InSb valence band - hence negative confinement energy
    Confined_valence_band_offsets = Strained_heavy_hole_band_offset - Valence_band_infinite_confined_levels
    '''

#_________________________________________________________________________#

#_________________________________________________________________________#

def finite_well_analytical_function(Energy, *Calculation_constants):
    #Extra parameters for calculation passed as touple!
    # touple is defined as:
    # *Calculation_constants[0] = Particle effective mass in barrier
    # *Calculation_constants[1] = Particle effective mass in quantum well
    # *Calculation_constants[2] = Finite quantum well energy barrier
    # *Calculation_constants[3] = Infinite quantum well ground state energy

    m_w = Calculation_constants[0]
    m_b = Calculation_constants[1]
    Vb = Calculation_cosntants[2]
    E1inf = Calculation_cosntants[3]

    return ( math.sqrt((m_w/m_b)*(Vb - Energy)) - math.sqrt(Energy)*math.tan((np.pi/2)*math.sqrt(Energy/E1inf)) )
#_________________________________________________________________________#

#_________________________________________________________________________#

def pretty_print_numpy_array(Array):
    for n in range (0, (Array.size)):
        print(Array[n])
#_________________________________________________________________________#

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
# Define main funtion

def main():
    '''Define constants for calculations'''
    #Array for mole Ga fraction for calculations
    Ga_mole_fraction = np.linspace(0,1,num=101)
    # temperature of crystal during measurement in Kelvin
    Measurment_temperature = 300
    Quantum_well_width = 100 #Angstroms!!

    '''Calculate In(1-x)Ga(x)As band parameters:'''
    #Calculate unstained band gap values:
    Unstrained_band_gap_energy_eV = calculate_alloy_band_gap(Ga_mole_fraction, Measurment_temperature)
    Unstrained_band_gap_energy_nm = convert_energy_to_wavelength(Unstrained_band_gap_energy_eV)

    #Calculate unstrained valence band energy offset (eV):
        #Note: Energy values are referenced to InSb valence band minimum (at 0 eV here)
    Unstrained_valence_Band_offset = calculate_valence_band_offset(Ga_mole_fraction)
    #Calculate conduction band offsets (eV):
        #Note: Valence band offset + band gap = conduction band
        #Note: Energy values are referenced to InSb valence band minimum (at 0 eV here)
    Unstrained_conduction_band_offset = Unstrained_valence_Band_offset + Unstrained_band_gap_energy_eV

    #calculate strain shifts in conduction band
    # Note: Eg(strained) = Eg(unstrained) + del Ec + P + Q
    #   -> strained conduction band = unstrained conduction band edge + del Ec
    #Calculate strained conduction band edge (referenced to InSb valence band)
    Del_Ec = calculate_alloy_del_Ec(Ga_mole_fraction, Measurment_temperature)
    Strained_conduction_band_offset = Unstrained_conduction_band_offset + Del_Ec

    #Calculat strain shifts to valence band edges
    #   Note: Eg(strained) = Eg(unstrained) + del Ec + P + Q
    #   Note: Strain shifts heavy hole and light hole bands in different directions
    #       -> Compressive strain shifts heavy hold band up -> quantum well confined states should be calculated for this band
    #       -> strained heavy hole band = unstrained valence band edge - P - Q
    #           -> Valnce band energy values are negative when referenced to InSb valence band
    P = calculate_alloy_P(Ga_mole_fraction, Measurment_temperature)
    Q = calculate_alloy_Q(Ga_mole_fraction, Measurment_temperature)
    Strained_heavy_hole_band_offset = Unstrained_valence_Band_offset - P - Q
    Strained_light_hole_band_offset = Unstrained_valence_Band_offset - P + Q

    #Calculate effective shifts in band edge from quantum confinement effects
    #Effective energy gap = Eg_strained + Ec_confinement + Ev_confinement
    Infinite_confined_levels = calculate_infinite_confined_levels(Ga_mole_fraction, Quantum_well_width, Measurment_temperature)
    Conduction_band_infinite_confined_levels = Infinite_confined_levels[0]
    Valence_band_infinite_confined_levels = Infinite_confined_levels[1]
    Confined_conduction_band_offsets = Strained_conduction_band_offset + Conduction_band_infinite_confined_levels
    #Valence band edge values are negative with respect to InSb valence band - hence negative confinement energy
    Confined_valence_band_offsets = Strained_heavy_hole_band_offset - Valence_band_infinite_confined_levels

    '''plot In(1-x)Ga(x)As band parameters:'''
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

    #fig4 generates a plot of the unstrained In(1-x)Ga(x)As gamma valley band gap as well as the strained gamma valley band gap
    #Note: Energy values are referenced to InSb valence band minimum (at 0 eV here)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1,1,1)
    ax4.set_ylabel('Unstrained In(1-x)Ga(x)As band gap (eV)')
    ax4.set_xlabel('Ga fraction')
    UVBE, = plt.plot(Ga_mole_fraction, Unstrained_valence_Band_offset, label="Unstrained In(1-x)Ga(x)As VB", color='b', linestyle='--')
    UCBE, = plt.plot(Ga_mole_fraction, Unstrained_conduction_band_offset, label="Unstrained In(1-x)Ga(x)As CB", color='r', linestyle='--')
    SHHBE, = plt.plot(Ga_mole_fraction,Strained_heavy_hole_band_offset, label="Strained In(1-x)Ga(x)As HHB", color='b')
    SLHBE, = plt.plot(Ga_mole_fraction,Strained_light_hole_band_offset, label="Strained In(1-x)Ga(x)As LHB", color='g')
    SCBE, = plt.plot(Ga_mole_fraction,Strained_conduction_band_offset, label="Strained In(1-x)Ga(x)As CB", color='r')
    #Unstrained_band_structure_legend = plt.legend(handles=[UVBE, UCBE, SHHBE, SLHBE, SCBE], loc=2)

    #fig4 generates a plot of the unstrained In(1-x)Ga(x)As gamma valley band gap as well as the strained gamma valley band gap
    #Note: Energy values are referenced to InSb valence band minimum (at 0 eV here)
    fig5 = plt.figure(5)
    ax5 = fig5.add_subplot(1,1,1)
    ax5.set_ylabel('Unstrained In(1-x)Ga(x)As effective band gap (eV)')
    ax5.set_xlabel('Ga fraction')
    UVBE, = plt.plot(Ga_mole_fraction, Unstrained_valence_Band_offset, label="Unstrained In(1-x)Ga(x)As VB", color='b', linestyle='--')
    UCBE, = plt.plot(Ga_mole_fraction, Unstrained_conduction_band_offset, label="Unstrained In(1-x)Ga(x)As CB", color='r', linestyle='--')
    SHHBE, = plt.plot(Ga_mole_fraction,Strained_heavy_hole_band_offset, label="Strained In(1-x)Ga(x)As HHB", color='b')
    SLHBE, = plt.plot(Ga_mole_fraction,Strained_light_hole_band_offset, label="Strained In(1-x)Ga(x)As LHB", color='g')
    SCBE, = plt.plot(Ga_mole_fraction,Strained_conduction_band_offset, label="Strained In(1-x)Ga(x)As CB", color='r')
    QCCBE, = plt.plot(Ga_mole_fraction, Confined_conduction_band_offsets, label="Strained In(1-x)Ga(x)As CB + QC energy", marker='*', fillstyle='none', color='r')
    QCVBE, = plt.plot(Ga_mole_fraction, Confined_valence_band_offsets, label="Strained In(1-x)Ga(x)As VB + QC energy", marker='*', fillstyle='none', color='b')
    plt.show()

#!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!---!!!#
# Start program
main()
