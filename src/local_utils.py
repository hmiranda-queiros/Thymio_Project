import math

def check_global_obstacles_and_global_path(length_advance, final_occupancy_grid, full_path, x_est):
    """
    Check if
     - there is an obstacle or
     - there is a map limit or
     - if it crossed the end of the path
    :param length_advance: distance of advancement to check
    :param final_occupancy_grid: map grid with obstacle size increase (1: obstacle, 0: no obstacle)
    :param full_path: list of the two coordinates of the optimal path before and after actual thymio's position
    :return: A boolean to know if there is an obstacle/map limit and a boolean to know if the A* path was crossed
    """

    # Initialization of the variables
    global_path = False
    obstacle = False
    DIST_ERROR = 20

    # Get the last position of the robot
    x = x_est[0][0]
    y = x_est[1][0]
    theta = x_est[2][0]
    conv = math.pi/180 # conversion deg to rad

    # Find the direction of advance to check the occupancy grid and global obstacles
    if (theta >= -22.5*conv) and (theta < 22.5*conv):  # 1
        dir_x = 1
        dir_y = 0
    elif (theta >= 22.5*conv) and (theta < 67.5*conv):  # 2
        dir_x = 1
        dir_y = 1
    elif (theta >= 67.5*conv) and (theta < 112.5*conv):  # 3
        dir_x = 0
        dir_y = 1
    elif (theta >= 112.5*conv) and (theta < 157.5*conv):  # 4
        dir_x = -1
        dir_y = 1
    elif ((theta >= 157.5*conv) and (theta <= 181*conv)) or ((theta >= -181*conv) and (theta < -157.5*conv)):  # 5
        dir_x = -1
        dir_y = 0
    elif (theta >= -157.5*conv) and (theta < -112.5*conv):  # 6
        dir_x = -1
        dir_y = -1
    elif (theta >= -112.5*conv) and (theta < -67.5*conv):  # 7
        dir_x = 0
        dir_y = -1
    elif (theta >= -67.5*conv) and (theta < -22.5*conv):  # 8
        dir_x = 1
        dir_y = -1
    else:
        dir_x = 0
        dir_y = 0
        #print(" angle is not between -180 and 180 degrees")

    # compute the next step
    x_next_step = x + length_advance * dir_x
    y_next_step = y + length_advance * dir_y

    # Verify the map limits and return obstacle=True if it reaches the limit
    if (x_next_step > 719) or (x_next_step < 1) or (y_next_step > 1279) or (y_next_step < 1):
        obstacle = True
        return obstacle, global_path

    # Verify the map obstacles in 2D and return obstacle=True if it reached it
    elif final_occupancy_grid[int(x_next_step)][int(y_next_step)] == 1:
        obstacle = True
        return obstacle, global_path

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
            return obstacle, global_path
            
    elif (abs(x-x_prev) < DIST_ERROR): # if the x is almost equal to the other two x's coordinates: we are again on the global path
        global_path = True
        return obstacle, global_path

    return obstacle, global_path
