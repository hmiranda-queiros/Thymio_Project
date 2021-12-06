import numpy as np
from motion import *

#Constant variances
q_x = 0.011  
q_y =  0.011  
q_theta = 0.0000087  
q_x_dot = 1.1  
q_y_dot =  1.1  
q_theta_dot = 0.00087


r_x = 0.025 
r_y = 0.18
r_theta = 0.00057
r_x_dot = 1.1  
r_y_dot = 1.1
r_theta_dot = 0.00087

Q = np.array([[q_x, 0, 0, 0, 0, 0], 
                [0, q_y, 0, 0, 0, 0], 
                [0, 0, q_theta, 0, 0, 0],
                [0, 0, 0, q_x_dot, 0, 0], 
                [0, 0, 0, 0, q_y_dot, 0], 
                [0, 0, 0, 0, 0, q_theta_dot]]);


#Camera is online
R_ON = ([[r_x, 0, 0, 0, 0, 0], 
        [0, r_y, 0, 0, 0, 0], 
        [0, 0, r_theta, 0, 0, 0],
        [0, 0, 0, r_x_dot, 0, 0], 
        [0, 0, 0, 0, r_y_dot, 0], 
        [0, 0, 0, 0, 0, r_theta_dot]]);
        
H_ON = np.array([[1, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]]);


#Camera is offline
R_OFF = ([[r_x_dot, 0, 0], 
        [0, r_y_dot, 0], 
        [0, 0, r_theta_dot]]);
        
H_OFF = np.array([[0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]]);
               
                
#Thymio goes forward                
A_F = np.array([[1, 0, 0, Ts, 0, 0], 
                [0, 1, 0, 0, Ts, 0], 
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]]);

#Thymio rotates
A_R = np.array([[1, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, Ts],
                [0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]]);

              
              
def kalman_filter(meas_pos, meas_speed_left, meas_speed_right, x_est_prev, P_est_prev, A, camera_on):
    
    
    ## Prediciton through the a priori estimate
    # estimated mean of the state
    x_est_a_priori = np.dot(A, x_est_prev) 
    
    # Estimated covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q
    
    
    #measurements
    speed_trans = (meas_speed_left + meas_speed_right) * thymio_speed_to_mms / 2
    speed_rot =  (meas_speed_right - meas_speed_left) * thymio_speed_to_rads / 2
    
    if camera_on :
        theta_meas = meas_pos[2]
        x_dot = speed_trans * np.cos(theta_meas)
        y_dot = speed_trans * np.sin(theta_meas)
        theta_dot = [speed_rot]
        
        y = np.array([meas_pos[0], meas_pos[1], meas_pos[2], x_dot, y_dot, theta_dot])
        H = H_ON
        R = R_ON
        
    
    else :
        theta_est = x_est_a_priori[2];
        x_dot = speed_trans * np.cos(theta_est)
        y_dot = speed_trans * np.sin(theta_est)
        theta_dot = [speed_rot]
        
        y = np.array([x_dot, y_dot, theta_dot])
        H = H_OFF
        R = R_OFF

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
    
        
       