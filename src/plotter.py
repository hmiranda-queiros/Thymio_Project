import math
import numpy as np
import matplotlib.pyplot as plt

    
def plot(ax1, ax2, ax3, x_est, P_est, fig):
    x1, y1 = gauss(x_est[0][0], P_est[0][0])
    x2, y2 = gauss(x_est[1][0], P_est[1][1])
    x3, y3 = gauss(x_est[2][0], P_est[2][2], 1)
    
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Gauss')
    
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('Gauss')
    
    ax3.set_xlabel('Theta (deg)')
    ax3.set_ylabel('Gauss')
    
    ax1.plot(x1, y1, 'g')
    ax2.plot(x2, y2, 'r')
    ax3.plot(x3, y3, 'y')
    fig.canvas.draw()

def gauss(mu, var, deg = 0):
    
    if deg == 0 :
        x = np.linspace(mu - 4 * math.sqrt(var), mu + 4 * math.sqrt(var), 200)
        
    else :
        var *= (180/math.pi)**2
        mu *= (180/math.pi)
        x = np.linspace(mu - 4 * math.sqrt(var),  mu + 4 * math.sqrt(var), 200)
        
    y = []
    
    for e in x:
        y.append(1/math.sqrt(2 * math.pi * var) * math.exp(-(e - mu)**2/(2 * var)))
        
    return x, y    
    
    
    