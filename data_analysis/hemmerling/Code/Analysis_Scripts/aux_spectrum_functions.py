import numpy as np
from constants import *
from energy_functions import get_energy, get_scaled_dunham



###########################################################################################

#def thermal_pop(T,J,v=0,iso=35):
#    μAlCl = __get_AlCl_mu(35)
#    # print(μAlCl)
#    BeX = 0.0
#    for i in range(0,4):
#        j = 1
#        BeX += __wtf(UklX[i,j]*μAlCl**(-(i+2*j)/2)*(v+1/2)**i)*1e-6 # MHz
#    # print(BeX)
#    kb = 2.083661912e4 # MHz*K^-1
#    Qrot = kb*T/(BeX) # check 2pi here
#    temp_pop = np.exp(-(BeX*J*(J+1))/(kb*T))/Qrot # and here
#    return temp_pop


def get_pop(v, J, we, B, Trot, Tvib):

    # returns population distribution

    Ewe = we * h_planck
    EB = B * h_planck

    beta_rot = 1/(kB*Trot)
    beta_vib = 1/(kB*Tvib)

    Ev = Ewe * (v+0.5)
    
    Erot = lambda J : EB * J * (J+1)

    # rotational distribution
    J_arr = np.arange(0, 200, 1)

    #print(Jarr)

    Nrot = np.sum( (2*J_arr+1) * np.exp(-beta_rot * Erot(J_arr)) )
    
    pop_rot = 1/Nrot * (2*J+1) * np.exp(-beta_rot * Erot(J))

    # vibrational distribution
    Nvib = np.exp(beta_vib * Ewe/2)/(np.exp(beta_vib * Ewe) - 1)

    pop_vib = 1/Nvib * np.exp(-beta_vib * Ev)

    return pop_vib * pop_rot


###########################################################################################

def get_doppler(T, f0):
    # returns Doppler width for 27Al35Cl

    return np.sqrt(8*kB*T*np.log(2)/((35+27)*amu*c**2)) * f0

###########################################################################################

def simple_gauss(x, x0, A, w):
    return A * np.exp( -(x-x0)**2/(2*w**2) )

def simple_lorentz(x, x0, A, w):
    return A * w**2/( (x-x0)**2 + w**2 )


###########################################################################################

def get_spectrum_from_cnts(x, cnts, a, T = 4, w = 0, ftype = 'gaussian'):

    y = np.zeros(len(x))

    for k in range(len(cnts)):

         if w == 0:
             w = get_doppler(T, cnts[k])

         if ftype == 'gaussian':
             y += simple_gauss(x, cnts[k], a[k], w)
         elif ftype == 'lorentz':
             y += simple_lorentz(x, cnts[k], a[k], w)

    return y

###########################################################################################

def get_line_centers(Yg, Ye, ve, vg, Jmax = 1, Trot = 10, Tvib = 100, real_amplitudes = True):

    Jg_arr = np.arange(0, Jmax+1)    
    Je_arr = np.arange(1, Jmax+1)
    
    f_P = []
    f_Q = []
    f_R = []
    
    A_P = []
    A_Q = []
    A_R = []
    
    for Jg in Jg_arr:
        for Je in Je_arr:
            
            # only dipole transitions
            eng = get_energy(Yg, Ye, vg, Jg, ve, Je)
            
            # apply population of ground states
            if real_amplitudes:
                A = get_pop(vg, Jg, 100*c*Yg[1][0], 100*c*Yg[0][1], Trot, Tvib)
            else:
                A = 1.0
            
            if Je - Jg == -1: 
                f_P.append(eng)
                A_P.append(A)
    
            if Je - Jg ==  0: 
                f_Q.append(eng)
                A_Q.append(A)
    
            if Je - Jg == +1: 
                f_R.append(eng)
                A_R.append(A)
    
    f_P = np.array(f_P)
    f_Q = np.array(f_Q)
    f_R = np.array(f_R)
    
    A_P = np.array(A_P)
    A_Q = np.array(A_Q)
    A_R = np.array(A_R)

    return ([f_P, f_Q, f_R], [A_P, A_Q, A_R])

#############################################################################################################

def get_spectrum(Ug = [], Ue = [], Dg = [], De = [], vg = 0, ve = 0, Jmax = 10, T = 10, df = 100e9, no_of_points = 5000):

    (Yg35, Ye35, Yg37, Ye37) = get_scaled_dunham(Ug, Ue, Dg, De)
   
    (lines35, ampl35) = get_line_centers(Yg35, Ye35, ve, vg, Jmax = Jmax, Trot = T, Tvib = 100*T)
    (lines37, ampl37) = get_line_centers(Yg37, Ye37, ve, vg, Jmax = Jmax, Trot = T, Tvib = 100*T)

    cnt_freq = np.mean(lines35[1])

    nus = np.linspace(cnt_freq - df, cnt_freq + df, no_of_points)
    
    g35 = {}
    g37 = {}

    g35['P'] = get_gaussians(nus, lines35[0], 0.76*ampl35[0], T = T)
    g37['P'] = get_gaussians(nus, lines37[0], 0.24*ampl37[0], T = T)

    g35['Q'] = get_gaussians(nus, lines35[1], 0.76*ampl35[1], T = T)
    g37['Q'] = get_gaussians(nus, lines37[1], 0.24*ampl37[1], T = T)

    g35['R'] = get_gaussians(nus, lines35[2], 0.76*ampl35[2], T = T)
    g37['R'] = get_gaussians(nus, lines37[2], 0.24*ampl37[2], T = T)



    return (nus, g35, g37)

#############################################################################################################

def get_transition_energies(Yg, Ye, vg, ve, Jmax = 1):

    Jg_arr = np.arange(0, Jmax+1)    
    Je_arr = np.arange(1, Jmax+1)
    
    lines_P = []
    lines_Q = []
    lines_R = []
    
    for Jg in Jg_arr:
        for Je in Je_arr:
            
            # only dipole transitions
            eng = get_energy(Yg, Ye, vg, Jg, ve, Je)
            
            if Je - Jg == -1: 
                lines_P.append([vg, Jg, ve, Je, eng])
    
            if Je - Jg ==  0: 
                lines_Q.append([vg, Jg, ve, Je, eng])
    
            if Je - Jg == +1: 
                lines_R.append([vg, Jg, ve, Je, eng])
    
    lines_P = np.array(lines_P)
    lines_Q = np.array(lines_Q)
    lines_R = np.array(lines_R)
    

    return (lines_P, lines_Q, lines_R)


