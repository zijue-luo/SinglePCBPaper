import matplotlib.pyplot as plt

import numpy as np

def gauss(x, x0, w, A):

    return A*w**2/( (x-x0)**2 + w**2 )



def get_Yb_freqs(offset = 0.0):

    # relative to Yb-174

    f1 = np.array([-508, -250, 0, 531, 589, 835, 1153, 1190, 1889]) + offset

    As = np.array([0.4, 0.1, 1, 0.7, 0.4, 0.38, 0.1, 0.1])

    return (f1, As)



def get_Rb_87_freqs(offset = 0.0):

    # relative to S1/2-P3/2
    # v = 384.230406373 THz
    
    cnt = 384.230484468

    f1 = np.array([+193.74, -72.9112, -229.8518]) - 2.563005979089e9/1e6 + offset

    As = np.array([1, 1, 1])

    return (f1, As, cnt)


def get_Rb_85_freqs(offset = 0.0):

    # relative to S1/2-P3/2
    # v = 384.230406373 THz
    
    cnt = 384.230406373

    f1 = np.array([+100.205, -20.435, -83.835, -113.208]) - 1.264888516e9/1e6 + offset

    As = np.array([1, 1, 1, 1])

    return (f1, As, cnt)



def get_W_freqs():

    cnt = 389.484510

    d46 = -12.7
    d42 = +10.0
    d43 = 10.0 - 4.5
    
    A = 47.1
    
    In = 0.5
    J = 1
    
    
    F = 3/2
    E3b = A/2 * (F * (F+1) - J*(J+1) - In*(In+1))
    
    F = 1/2
    E3a = A/2 * (F * (F+1) - J*(J+1) - In*(In+1))
 
    f1 = np.array([0, d46, d42, d43 + E3b, d43 + E3a])

    As = [500, 400, 300, 100, 100]

    return (f1, As, cnt)


def get_W(w = 5.0):
    
   
    (fs, As) = get_W_freqs()

    
    x = np.linspace(-100, 100, 1000)
    
    y = 0 * x
 
    for k in range(len(fs)):
        y += gauss(x, fs[k], w, As[k])
    
    y = y/max(y)

    return (x, y)


def get_K_freqs():

    f1 = np.array([14.4, -6.7, -16.1, -19.4]) - 173.1
    f2 = np.array([14.4, -6.7, -16.1, -19.4]) + 288.6

    return np.append(f1, f2)


def get_K(w = 5.0):

    x = np.linspace(-1000, 1000, 5000)
    
    y = 0 * x
    
    fs = get_K_freqs()
   
    for f in fs:
        y += gauss(x, f , w, 1)
    
    y = y/max(y)



    return (x, y)


if __name__ == '__main__':

    get_K()

    w0 = 7.79
    w1 = 6.0
    w2 = 4.0
    
    (x, y) = get_W(w = w0)
    
    plt.plot(x, y, label = w0)
    
    plt.legend()
    
    
    plt.show()







