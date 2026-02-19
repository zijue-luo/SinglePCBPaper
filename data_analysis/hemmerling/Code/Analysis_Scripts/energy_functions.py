import numpy as np
import sys

from constants import *

###########################################################################################

def scale_dunham_matrix(M, D, isotope, scale = True):

    # scale or unscale Dunham coefficients M with the reduced mass and the Born-Oppenheimer correction D

    if scale == False:
        
        # return U_kl = 1/mu^(-k/2 - l) * M_kl
        U = []
        for k in range(len(M)):
            hlp = []
            for l in range(len(M[k])):
                hlp.append( 1.0/scale_dunham(k, l, isotope, bob_corr = D[k][l]) * M[k][l] )
            U.append(hlp)
 
        return U
    else:

        # return Y_kl = mu^(-k/2 - l) * M_kl
        Y = []
        for k in range(len(M)):
            hlp = []
            for l in range(len(M[k])):
                #hlp.append( mu**(-k/2.0 - l) * M[k][l] )
                hlp.append( scale_dunham(k, l, isotope, bob_corr = D[k][l]) * M[k][l] )
            Y.append(hlp)
        
        return Y

###########################################################################################

def get_scaled_dunham(Ug, Ue, Dg, De):

    # ground state
    Yg35 = scale_dunham_matrix(Ug, Dg, 35, scale = True)
    Yg37 = scale_dunham_matrix(Ug, Dg, 37, scale = True)

    # excited state
    Ye35 = scale_dunham_matrix(Ue, De, 35, scale = True)
    Ye37 = scale_dunham_matrix(Ue, De, 37, scale = True)

    return (Yg35, Ye35, Yg37, Ye37)


#########################################################################

def scale_dunham(k, l, isotope, bob_corr = 0.0):
    
    # returns the scaling coefficient Y = coeff * U

    mass1 = massAl # aluminum
    fac = 1.0

    if isotope == 35:
        mass2 = massCl_35
    elif isotope == 37:
        mass2 = massCl_37
    else:
        print('Wrong isotope')


    # Add Born-Oppenheimer breakdown
    fac = (1 + mass_e/(mass2 * amu) * bob_corr)

    #if not bob_corr == 0.0 and (k == 0 and l == 0):
    #    print('Bob: {0}'.format(bob_corr))
    #    print(isotope)
    #    print(k)
    #    print(l)
    #    print(fac - 1)

    mu = (mass1 * mass2)/(mass1 + mass2)


    return mu**(-k/2.0 - l) * fac


########################################################################

def get_energy(Yg, Ye, vg, Jg, ve, Je):
    
    return energy(Ye, ve, Je, 1) - energy(Yg, vg, Jg, 0)

########################################################################

def energy(Y, v, J, L):

    e = 0.0
    for k in range(len(Y)):
        for l in range(len(Y[k])):

            e += calc_energy(Y[k][l], v, J, k, l, L)

    return e

#########################################################################

def calc_energy(val, v, J, k, l, L):

    return 100 * c * (val * (v + 0.5)**k * ( J * (J + 1.0) - L**2 )**l)

#########################################################################

def get_params_energy(params, vg, Jg, ve, Je, isotope):
    # retrieves the energy of a transition from the fit parameters

    e = 0.0
    for key in params.keys():

        # e.g. Ue35
        k = int(key[2])
        l = int(key[3])

        state = key[0:2]
        
        # scale mass-reduced coefficients to the correct isotope
        
        if state == 'Ue':
        
            val = scale_dunham(k, l, isotope, bob_corr = params['De' + str(k) + str(l)].value) * params[key].value
            e += calc_energy(val, ve, Je, k, l, 1)

        if state == 'Ug':
        
            val = scale_dunham(k, l, isotope, bob_corr = params['Dg' + str(k) + str(l)].value) * params[key].value
            e -= calc_energy(val, vg, Jg, k, l, 0)

    return e

#########################################################################


