import math
import numpy as np
import matplotlib.pyplot as plt

# Thymio outline
center_offset = np.array([5.5,5.5])
thymio_coords = np.array([[0,0], [11,0], [11,8.5], [10.2, 9.3], 
                          [8, 10.4], [5.5,11], [3.1, 10.5], 
                          [0.9, 9.4], [0, 8.5], [0,0]])-center_offset

def rotate(angle, coords):
    """
    Rotates the coordinates of a matrix by the desired angle
    :param angle: angle in radians by which we want to rotate
    :return: numpy.array() that contains rotated coordinates
    """
    R = np.array(((np.cos(angle), -np.sin(angle)),
                  (np.sin(angle),  np.cos(angle))))
    
    return R.dot(coords.transpose()).transpose()
    
def start_plot(x_est, P_ext):
    abs_Thymio_coords = thymio_coords + np.array([x_est[-1][1][0], x_est[-1][0][0]])
    x, y = gauss(x_est[-1][0][0], P_ext[-1][0][0])

    plt.figure(figsize=(10,20))
    plt.subplot(211)
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.gca().invert_yaxis()
    body, = plt.plot(abs_Thymio_coords[:, 0], abs_Thymio_coords[:, 1], color="g")
    
    plt.subplot(212)
    g_x, = plt.plot(x, y, color="b")
        
    return body  

def gauss(mu, sigma):
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 50)
    y = []
    
    for e in x:
        y.append(1/(math.sqrt(2 * math.pi) * sigma) * math.exp(-(e - mu)**2/(2 * sigma**2)))

    return x, y    
    
def update_plot(x_est, body):
    rotated_thymio_coords = rotate(- x_est[-1][2][0], thymio_coords) 
    abs_Thymio_coords = rotated_thymio_coords + np.array([x_est[-1][1][0], x_est[-1][0][0]])
    
    body.set_xdata(abs_Thymio_coords[:, 0])
    body.set_ydata(abs_Thymio_coords[:, 1])
    plt.draw()