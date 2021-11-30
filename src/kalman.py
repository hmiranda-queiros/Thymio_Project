import numpy as np
from threading import Timer
import math



#Constants
Ts = 0.1
thymio_speed_to_mms = 0.341
distance_wheel = 95

#Thymio goes forward
q_F_x = 0.008
q_F_y =  0.008
q_F_theta = 0.000006
q_F_x_dot = 0.779  
q_F_y_dot =  0.779  
q_F_theta_dot = 0.000591

Q_F = np.array([[q_F_x, 0, 0, 0, 0, 0], 
                [0, q_F_y, 0, 0, 0, 0], 
                [0, 0, q_F_theta, 0, 0, 0],
                [0, 0, 0, q_F_x_dot, 0, 0], 
                [0, 0, 0, 0, q_F_y_dot, 0], 
                [0, 0, 0, 0, 0, q_F_theta_dot]]);

r_F_x_dot = 0.779  
r_F_y_dot = 0.779
r_F_theta_dot = 0.000591

R_F = np.array([[r_F_x_dot, 0, 0],
               [0, r_F_y_dot, 0],
               [0, 0, r_F_theta_dot]]);

A_F = np.array([[1, 0, 0, Ts, 0, 0], 
                [0, 1, 0, 0, Ts, 0], 
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]]);


#Thymio rotates
q_R_x = 0.01
q_R_y =  0.01
q_R_theta = 0.00001 
q_R_x_dot = 1.31
q_R_y_dot =  1.31
q_R_theta_dot = 0.00122

Q_R = np.array([[q_R_x, 0, 0, 0, 0, 0], 
                [0, q_R_y, 0, 0, 0, 0], 
                [0, 0, q_R_theta, 0, 0, 0],
                [0, 0, 0, q_R_x_dot, 0, 0], 
                [0, 0, 0, 0, q_R_y_dot, 0], 
                [0, 0, 0, 0, 0, q_R_theta_dot]]);

r_R_x_dot = 1.31
r_R_y_dot = 1.31
r_R_theta_dot = 0.00122  

R_R = np.array([[r_R_x_dot, 0, 0],
               [0, r_R_y_dot, 0],
               [0, 0, r_R_theta_dot]]);

A_R = np.array([[1, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, Ts],
                [0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]]);

# commun matrix
H = np.array([[0, 0, 0, 1, 0, 0], 
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]]);
              
              
def kalman_filter(meas_speed_left, meas_speed_right, x_est_prev, P_est_prev, Q, R, A):
    
    
    ## Prediciton through the a priori estimate
    # estimated mean of the state
    x_est_a_priori = np.dot(A, x_est_prev) 
    
    # Estimated covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q
    
    
    #measurements
    speed_trans = (meas_speed_left + meas_speed_right) * thymio_speed_to_mms / 2
    speed_rot =  (meas_speed_right - meas_speed_left) * (thymio_speed_to_mms / distance_wheel)
    
    theta_est = x_est_a_priori[2];
    x_dot = speed_trans * math.cos(theta_est)
    y_dot = speed_trans * math.sin(theta_est)
    theta_dot = speed_rot
    
    y = np.array([[x_dot], [y_dot], [theta_dot]])
   

    # innovation / measurement residual
    i = y - np.dot(H, x_est_a_priori)
    # measurement prediction covariance
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
             
    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))
    
    
    # a posteriori estimate
    x_est = x_est_a_priori + np.dot(K,i)
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori))
    
    return x_est, P_est
    
    
class RepeatedTimer(object):
    def __init__(self, interval, function, *args):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False