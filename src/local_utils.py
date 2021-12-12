import math
import numpy as np

import math
def check_obstacles(final_occupancy_grid, x_est):
    """
    Check if there is an expected obstacle in front of Thymio or if there is a map limit
    :param final_occupancy_grid: map grid with obstacle size increase (1: obstacle, 0: no obstacle)
    :param x_est: pose of Thymio from Kalman
    :return obstacle: boolean asserting the existence of unexpected obstacle (1:yes 0:no)
    """

    # Initialization of the variables
    obstacle = False

    # Get the last position of the robot
    x = x_est[0][0]
    y = x_est[1][0]
    theta = x_est[2][0]
    conv = math.pi/180 # conversion deg to rad
    #radius of arc with respect to robot
    dist_to_check = list(range(0,300,20))
    #direction with respect to robot front direction
    sensor_angles = [ele*conv for ele in list(range(-90,90,10))]
    dx_obs=[]
    dy_obs=[]
    for d in dist_to_check:
        for th in sensor_angles:
            x_tmp = x+d * np.cos(th+theta)
            y_tmp = y+d * np.sin(th+theta)
            #check if the half-disc is outside the map
            if x_tmp < 720 and x_tmp>=0 and y_tmp < 1280 and y_tmp>=0:
                dx_obs.append(x_tmp)
                dy_obs.append(y_tmp)
    #retrieve all values from occupancy grid
    obs_coords_temp = [[int(dx_obs), int(dy_obs)] for (dx_obs, dy_obs) in zip(dx_obs, dy_obs)]
    
    for coord in obs_coords_temp:
        if final_occupancy_grid[coord[0],coord[1]] == 1:
            obstacle = True
            break
    #retrieve all values from occupancy grid
    #set obstacle and return if any of the grid is occupied
    return obstacle

def check_returned_to_global_path(full_path, x_est):
    """
    Check if Thymio back to global path in local avoidance mode
    :param full_path: list of the two coordinates of the optimal path before and after actual thymio's position
    :param x_est: pose of Thymio from Kalman
    :return: A boolean to know if the A* path was crossed
    
    """
    # Initialization of the variables
    global_path = False
    DIST_ERROR = 20

    # Get the last position of the robot
    x = x_est[0][0]
    y = x_est[1][0]

    # Compute if we reach the global path (with an error margin)
    # First, let's get the previous and the next coordinates of the full path wrt the actual position of the robot
    
    x_prev = full_path[0][1][0]
    y_prev = full_path[0][1][1]
    x_next = full_path[1][1][0]
    y_next = full_path[1][1][1]

    # Let's now compute the straight line between the previous vertices and the following ones
    if (x_next-x_prev) != 0:
        slope = (y_next - y_prev) / (x_next - x_prev) # slope
        intercept = y_prev - slope * x_prev # intercept
        evaluation = abs(slope * x + intercept - y) # check if we reached the global path
        if evaluation < DIST_ERROR:  # margin of error in mm
            global_path = True
    elif (abs(x-x_prev) < DIST_ERROR): # if the x is almost equal to the other two x's coordinates: we are again on the global path
        global_path = True
    return  global_path

