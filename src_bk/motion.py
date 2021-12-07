import numpy as np
from threading import Timer
import math


#Constants
Ts = 0.1
thymio_speed_to_mms = 0.36
thymio_speed_to_rads = 0.0071


def move_forward(dist):
    time = abs(dist) / (thymio_speed_to_mms * 100)
    size_command = int(time / Ts)
    commands = []
    
    for i in range(size_command) :
        commands.append([100, 100])
    
    return commands
    
def move_backward(dist):
    time = abs(dist) / (thymio_speed_to_mms * 100)
    size_command = int(time / Ts)
    commands = []
    
    for i in range(size_command) :
        commands.append([-100, -100])
    
    return commands
    
def turn_left(angle):
    time = abs(angle) / (thymio_speed_to_rads * 100)
    size_command = int(time / Ts)
    commands = []
    
    for i in range(size_command) :
        commands.append([-100, 100])
    
    return commands
    
def turn_right(angle):
    time = abs(angle) / (thymio_speed_to_rads * 100)
    size_command = int(time / Ts)
    commands = []
    
    for i in range(size_command) :
        commands.append([100, -100])
    
    return commands