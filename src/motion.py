import math


# Kalman global variables
Ts = 0.1
thymio_speed_to_mms = 0.36
thymio_speed_to_rads = 0.0071


def move_forward(dist):
    """
    Calculates a set of motor commands which enables thymio to move forward for the desired amount
    :param dist: the desired amount of linear movement in mm [a float]
    :return: set of motor commands to give every Ts seconds [list of 2-element inner lists]
    """
    global thymio_speed_to_mms, Ts
    time = abs(dist) / (thymio_speed_to_mms * 100)
    size_command = math.ceil(time / Ts)
    commands = []
    for i in range(size_command):
        commands.append([100, 100])
    return commands


def move_backward(dist):
    """
    Calculates a set of motor commands which enables thymio to move backward for the desired amount
    :param dist: the desired amount of linear movement in mm [a float]
    :return: set of motor commands to give every Ts seconds [list of 2-element inner lists]
    """
    global thymio_speed_to_mms, Ts
    time = abs(dist) / (thymio_speed_to_mms * 100)
    size_command = math.ceil(time / Ts)
    commands = []
    for i in range(size_command):
        commands.append([-100, -100])
    return commands


def turn_left(angle):
    """
    Calculates a set of motor commands which enables thymio to rotate left for the desired amount
    :param angle: the desired amount of rotation in radians [a float]
    :return: set of motor commands to give every Ts seconds [list of 2-element inner lists]
    """
    global thymio_speed_to_rads, Ts
    time = abs(angle) / (thymio_speed_to_rads * 100)
    size_command = math.ceil(time / Ts)
    commands = []
    for i in range(size_command):
        commands.append([-100, 100])
    return commands


def turn_right(angle):
    """
    Calculates a set of motor commands which enables thymio to rotate right for the desired amount
    :param angle: the desired amount of rotation in radians [a float]
    :return: set of motor commands to give every Ts seconds [list of 2-element inner lists]
    """
    global thymio_speed_to_rads, Ts
    time = abs(angle) / (thymio_speed_to_rads * 100)
    size_command = math.ceil(time / Ts)
    commands = []
    for i in range(size_command):
        commands.append([100, -100])
    return commands
