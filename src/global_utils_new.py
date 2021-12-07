import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
from scipy.interpolate import interp1d
import math
from motion import *

def convert_to_np(obs_coords, start, goal):
    """
    Converts the coordinates of obstacles, start, and goal to numpy array
    :param obs_coords: obstacle coordinates [list of lists of inner lists]
    :param start: start coordinates [tuple]
    :param goal: goal coordinates [tuple]
    :return: obs_coords [list of numpy 2d arrays], start [numpy 1d array], and goal [numpy 1d array]
    """
    start = np.squeeze(np.array(start))
    goal = np.squeeze(np.array(goal))
    for i in range(len(obs_coords)):
        obs_coords[i] = np.array(obs_coords[i])
    return obs_coords, start, goal


def sort_obs_coords(obs_coords, clockwise=True):
    """
    Sorts the coordinates of obstacles in a clockwise or counterclockwise order
    :param obs_coords: obstacle coordinates [list of numpy 2d arrays]
    :param clockwise: [boolean] | True (sort clockwise), False (sort counterclockwise)
    :return: sorted version of obs_coords [list of numpy 2d arrays]
    """
    for i in range(len(obs_coords)):
        obs = obs_coords[i]
        cx = np.mean(obs[:, 0])
        cy = np.mean(obs[:, 1])
        angles = [np.arctan2(x-cx, y-cy) for (x, y) in obs]
        indices = sorted(range(len(angles)), key=angles.__getitem__)
        if clockwise:
            obs = [obs[idx] for idx in indices]
            obs_coords[i] = np.array(obs)
        else:
            obs = [obs[idx] for idx in indices[::-1]]
            obs_coords[i] = np.array(obs)
    return obs_coords


def swap_xy(obs_coords, start, goal, edges, optimal_path):
    """
    Swaps the x and y coordinates to be able to plot the results in our desired reference coordinate system
    :param obs_coords: obstacle coordinates [list of numpy 2d arrays]
    :param start: start coordinates [numpy 1d array]
    :param goal: goal coordinates [numpy 1d array]
    :param edges: edges of the visibility graph [list of lists]
    :param optimal_path: optimal path derived by A* [list of lists]
    :return: obs_coords_swap [list of numpy 2d arrays], start_swap [numpy 1d array], goal_swap [numpy 1d array],
             edges_swap (only set of the coordinates of vertices) [list of lists],
             optimal_path_swap (only the coordinates of vertices) [list of lists]
    """
    # Start
    start_swapped = np.copy(start)
    start_swapped[1], start_swapped[0] = start_swapped[0], start_swapped[1]
    # Goal
    goal_swapped = np.copy(goal)
    goal_swapped[1], goal_swapped[0] = goal_swapped[0], goal_swapped[1]
    # Obstacles
    obs_coords_swapped = []
    for i in range(len(obs_coords)):
        obs_coords_swapped.append(np.copy(obs_coords[i]))
        obs_coords_swapped[i][:, [1, 0]] = obs_coords_swapped[i][:, [0, 1]]
    # Edges
    edges_swapped = []
    for i in range(len(edges)):
        temp1 = np.copy(edges[i][1])
        temp1[1], temp1[0] = temp1[0], temp1[1]
        temp2 = np.copy(edges[i][3])
        temp2[1], temp2[0] = temp2[0], temp2[1]
        edges_swapped.append([temp1, temp2])
    # Optimal Path
    optimal_path_swapped = []
    for i in range(len(optimal_path)):
        temp = np.copy(optimal_path[i][1])
        temp[1], temp[0] = temp[0], temp[1]
        optimal_path_swapped.append(temp)
    return obs_coords_swapped, start_swapped, goal_swapped, edges_swapped, optimal_path_swapped


def plot_map(obs_coords, start, goal, edges, optimal_path):
    """
    Plots the map
    :param obs_coords: obstacle coordinates (swapped) [list of numpy 2d arrays]
    :param start: start coordinates (swapped) [numpy 1d array]
    :param goal: goal coordinates (swapped) [numpy 1d array]
    :param edges: only set of vertices (swapped) [list of lists]
    :param optimal_path: optimal path (swapped) (only the coordinates of vertices) [list of lists]
    :param x_size: the length of the x coordiante of the reference system
    :param y_size: the length of the y coordinate of the reference system
    :return: None
    """
    # Global Variables
    x_size = 720
    y_size = 1280
    # Options
    x_step = 80
    y_step = 160
    plt.figure(1)
    plt.grid(True)
    plt.axis('square')
    plt.axis([0, y_size, 0, x_size])
    plt.xticks(np.arange(0, y_size+y_step, step=y_step))
    plt.yticks(np.arange(0, x_size+x_step, step=x_step))
    plt.setp(plt.gca(), 'ylim', reversed(plt.getp(plt.gca(), 'ylim')))
    plt.gca().spines['bottom'].set_position('zero')
    plt.gca().tick_params(axis="x", direction="in", labeltop="on", labelbottom=False)
    plt.gca().patch.set_edgecolor('black')
    plt.gca().patch.set_linewidth('0.5')
    # Start
    plt.plot(start[0], start[1], marker='.', markerfacecolor='#7E2F8E', markeredgecolor='#7E2F8E', markersize=15)
    # Goal
    plt.plot(goal[0], goal[1], marker='.', markerfacecolor='#7E2F8E', markeredgecolor='#7E2F8E', markersize=15)
    # Obstacles
    count = 2
    for obs in obs_coords:
        for i in range(obs.shape[0]):
            if i == obs.shape[0] - 1:
                plt.plot([obs[0, 0], obs[-1, 0]], [obs[0, 1], obs[-1, 1]], '-', color='#0072BD',
                         marker='.', markerfacecolor='#D95319', markeredgecolor='#D95319', markersize=10)
                plt.text(obs[-1, 0], obs[-1, 1], str(count))
                count = count + 1
                break
            plt.plot([obs[i, 0], obs[i+1, 0]], [obs[i, 1], obs[i+1, 1]], '-', color='#0072BD',
                     marker='.', markerfacecolor='#D95319', markeredgecolor='#D95319', markersize=10)
            plt.text(obs[i, 0], obs[i, 1], str(count))
            count = count + 1
    # Edges
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k--', linewidth=0.5)
    # Optimal Path
    for i in range(len(optimal_path)-1):
        plt.plot([optimal_path[i][0], optimal_path[i+1][0]], [optimal_path[i][1], optimal_path[i+1][1]],
                 '-', color='#77AC30')
    plt.text(start[0], start[1], '1')
    plt.text(goal[0], goal[1], str(count))
    plt.savefig('map.png')


def print_data(nodes, edges, h, optimal_path, discretized_path):
    for item in locals().items():
        print(item[0])
        print(len(item[1]))
        for element in item[1]:
            print(element)
        print('------------------------------------------------------------')


def create_obs_shape(obs_coords):
    obs_shapes = []
    for obs in obs_coords:
        # obs = np.squeeze(obs)
        obs_shapes.append(sg.Polygon(obs))
    return obs_shapes


def is_node_reachable(node, map_margin = 20, x_size = 720, y_size = 1280):
    x_lb = map_margin
    x_ub = x_size - map_margin
    y_lb = map_margin
    y_ub = y_size - map_margin
    node_x = node[2][0]
    node_y = node[2][1]
    if node_x <= x_lb or node_x >= x_ub:
        return False
    elif node_y <= y_lb or node_y >= y_ub:
        return False
    return True


def vis_graph(obs_coords, start, goal):
    # Number of nodes
    num_nodes = 2
    for obs in obs_coords:
        num_nodes = num_nodes + obs.shape[0]
    # Nodes
    nodes = [[1, 0, start, 0]]
    j = 2
    for i in range(len(obs_coords)):
        obs = obs_coords[i]
        for vrtc in obs:
            nodes.append([j, i+1, vrtc, len(obs)])
            j = j + 1
    nodes.append([num_nodes, len(obs_coords)+1, goal, 0])
    # Edges
    edges = []
    for i in range(num_nodes):
        if not is_node_reachable(nodes[i]):
            continue
        for j in range(i+1, num_nodes):
            if not is_node_reachable(nodes[j]):
                continue
            elif (j - i > 1) and (nodes[j][1] == nodes[i][1]):    # Skipping Diagonals
                continue
            elif (j - i == 1) and (nodes[j][1] - nodes[i][1] == 0):  # Adding Exteriors
                edges.append([nodes[i][0], nodes[i][2], nodes[j][0], nodes[j][2], np.linalg.norm(nodes[i][2] - nodes[j][2])])
            else:   # Skipping Collisions with Obstacles and Collinear Edges
                if (j - i == 1) and (nodes[j][1] - nodes[i][1] == 1) and (nodes[i][1] != 0):  # Adding Last Obstacle Edge
                    temp = i - nodes[i][3] + 1
                    edges.append([nodes[temp][0], nodes[temp][2], nodes[i][0], nodes[i][2], np.linalg.norm(nodes[temp][2] - nodes[i][2])])
                line = sg.LineString(np.vstack((nodes[j][2], nodes[i][2])))
                obs_shapes = create_obs_shape(obs_coords)
                check1 = [line.crosses(obs) for obs in obs_shapes]
                flag = 0
                for obs in obs_shapes:
                    check2 = [line.contains(sg.Point(coord)) for coord in obs.boundary.coords]
                    if any(check2):
                        flag = 1
                        break
                if flag == 1:
                    continue
                if not any(check1):
                    edges.append([nodes[i][0], nodes[i][2], nodes[j][0], nodes[j][2], np.linalg.norm(nodes[i][2]-nodes[j][2])])
    edges.sort(key=lambda x: (x[0], x[4]))
    return nodes, edges


def heuristic(nodes):
    goal = nodes[-1][2]
    h = []
    for i in range(len(nodes)):
        h.append([nodes[i][0], np.linalg.norm(goal-nodes[i][2])])
    return h


def generate_path(path_so_far, current, nodes):
    optimal_path = [[current, nodes[current-1][2]]]
    while current in path_so_far.keys():
        # Add where the current node came from to the start of the list
        optimal_path.insert(0, [path_so_far[current], nodes[path_so_far[current]-1][2]])
        current = path_so_far[current]
    return optimal_path


def a_star(nodes, edges, h):
    start = nodes[0][0]
    goal = nodes[-1][0]
    past_cost = {start: 0}    # g
    total_cost = {start: past_cost[start] + h[0][1]}    # f
    open_set = [start]
    closed_set = []
    path_so_far = {}
    for i in range(1, len(nodes)):
        past_cost[nodes[i][0]] = np.inf
        total_cost[nodes[i][0]] = np.inf
    while open_set != []:
        total_cost_open = {key: val for (key, val) in total_cost.items() if key in open_set}
        current = min(total_cost_open, key=total_cost_open.get)
        del total_cost_open
        if current == goal:
            return generate_path(path_so_far, current, nodes)
        open_set.remove(current)
        closed_set.append(current)
        neighbors1 = [[edge[2], edge[4]] for edge in edges if edge[0] == current]
        neighbors2 = [[edge[0], edge[4]] for edge in edges if edge[2] == current]
        neighbors = neighbors1 + neighbors2
        neighbors.sort(key=lambda x: (x[0], x[1]))
        for neighbor in neighbors:
            if neighbor[0] in closed_set:
                continue
            tentative_past_cost = past_cost[current] + neighbor[1]
            if neighbor[0] not in open_set:
                open_set.append(neighbor[0])
            if tentative_past_cost < past_cost[neighbor[0]]:
                path_so_far[neighbor[0]] = current
                past_cost[neighbor[0]] = tentative_past_cost
                total_cost[neighbor[0]] = past_cost[neighbor[0]] + h[neighbor[0]-1][1]
                # total_cost[neighbor[0]] = past_cost[neighbor[0]]
    print('Failed to find a path to the goal!')
    return []


def calc_angle(p1, p2):
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    if angle < 0:
        angle = angle + 2 * np.pi
    return angle


def discretize_path(start_po, edges, optimal_path):
    movements = []
    for i in range(len(optimal_path)-1):
        nodei0 = optimal_path[i]
        nodei1 = optimal_path[i+1]
        angle = calc_angle(nodei0[1], nodei1[1])
        movements.append(angle)
        dist = [edge[4] for edge in edges if ((edge[0] == nodei0[0]) and (edge[2] == nodei1[0])) or ((edge[2] == nodei0[0]) and (edge[0] == nodei1[0]))][0]
        movements.append(dist)

    discretized_path = []
    discretized_path.append([1, movements[0] - start_po[-1], 1, np.array(start_po[0:-1]), start_po[-1],\
                             optimal_path[1][0], optimal_path[1][1], movements[0]])
    for i in range(1, len(movements)):
        idx = i // 2
        if i % 2 == 1:
            discretized_path.append([2, movements[i], optimal_path[idx][0], optimal_path[idx][1], movements[i-1],\
                                     optimal_path[idx+1][0], optimal_path[idx+1][1], movements[i-1]])
        else:
            discretized_path.append([1, movements[i] - movements[i-2], optimal_path[idx][0], optimal_path[idx][1],\
                                     movements[i-2], optimal_path[idx+1][0], optimal_path[idx+1][1], movements[i]])
    return discretized_path


def calibrate_path(current_po, node):
    angle = calc_angle(current_po[0:-1], node)
    angle = angle - current_po[-1]
    dist = np.linalg.norm(node - current_po[0:-1])
    return angle, dist


def is_target_reached(target_po, current_po, TH_EPS_RO, TH_EPS_DI, commands):
    if (len(commands) == 0) or \
       ((np.linalg.norm(target_po[:-1] - np.array(current_po[:-1])) <= TH_EPS_DI) and\
       (np.abs(target_po[-1]-current_po[-1]) <= TH_EPS_RO)):
        return True
    else:
        return False


def set_motors_speed(motor_speeds):
    return {
        'motor.left.target': [int(motor_speeds[0])],
        'motor.right.target': [int(motor_speeds[1])],
    }


def convert_sensor_val_to_mm(val):
    global sensor_vals, real_dists
    if val == 0:
        return np.inf
    f = interp1d(sensor_vals, real_dists, kind='linear')
    return f(val).item()


def convert_mm_to_sensor_val(val):
    global sensor_vals, real_dists
    if val == 0:
        return np.inf
    f = interp1d(real_dists, sensor_vals, kind='linear')
    return f(val).item()


def get_obs_coords_from_sensor_vals(vals, current_po):
    global sensor_angles, sensor_po
    current_x = current_po[0]
    current_y = current_po[1]
    current_angle = current_po[-1]
    dist_to_sensor = [convert_sensor_val_to_mm(val) for val in vals]
    dx_obs = [d * np.cos(th+current_angle) for (d, th) in zip(dist_to_sensor, sensor_angles)]
    dy_obs = [d * np.sin(th+current_angle) for (d, th) in zip(dist_to_sensor, sensor_angles)]
    dx_sensor = [item[0] * np.cos(item[1]) for item in sensor_po]
    dy_sensor = [item[0] * np.sin(item[1]) for item in sensor_po]
    obs_coords_temp = [[current_x+dx_o+dx_s, current_y+dy_o+dy_s] for (dx_o, dy_o, dx_s, dy_s) in zip(dx_obs, dy_obs, dx_sensor, dy_sensor)]
    return np.array(obs_coords_temp)


def calc_avg_angle(vals):
    global sensor_angles
    max_index = np.argmax(vals)
    if max_index == 0 or max_index == 1:
        num = [vals[i] * sensor_angles[i] for i in range(0, 3)]
        den = np.sum(vals[0:3])
    elif max_index == 2:
        num = [vals[i] * sensor_angles[i] for i in range(1, 4)]
        den = np.sum(vals[1:4])
    elif max_index == 3 or max_index == 4:
        num = [vals[i] * sensor_angles[i] for i in range(2, 5)]
        den = np.sum(vals[2:5])
    avg_angle = sum(num) / den
    return avg_angle    # radian


def correct_path(current_po, node1, node2):
    # node1 and node2 [numpy 1d array]
    # current_po = [x, y, theta]
    line_angle = calc_angle(node1, node2)
    p1 = node1
    p2 = node2
    p3 = np.array(current_po[0:-1])
    distance = np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
    return line_angle, distance


def get_commands(thym_state, value):
    if thym_state == 2 and value >= 0:      # FORWARD +
        return move_forward(value)
    elif thym_state == 2 and value < 0:     # FORWARD -
        return move_backward(value)
    elif thym_state == 1 and value <= 0:    # ROTATE Right
        return turn_right(value)
    elif thym_state == 1 and value > 0:     # ROTATE Left
        return turn_left(value)
    return


def calc_commands(discretized_path):
    commands = []
    for item in discretized_path:
        commands.extend(get_commands(item[0], item[1]))
    return commands


def do_global_navigation(obs_coords, start_po, goal_po):
    # [obs_coords, start_po, goal_po] = import_from_vision()
    [obs_coords, start, goal] = convert_to_np(obs_coords, start_po[0:-1], goal_po[0:-1])
    obs_coords = sort_obs_coords(obs_coords)
    [nodes, edges] = vis_graph(obs_coords, start, goal)
    h = heuristic(nodes)
    optimal_path = a_star(nodes, edges, h)
    discretized_path = discretize_path(start_po, edges, optimal_path)
    # plot_map(*swap_xy(obs_coords, start, goal, edges, optimal_path))
    # print_data(nodes, edges, h, optimal_path, discretized_path)
    return discretized_path, optimal_path


def get_nodes_of_commands(discretized_path, commands):
    nodes_of_commands = []
    j = 0
    for i in range(len(commands) - 1):
        nodes_of_commands.append([discretized_path[j][3], discretized_path[j][6]])
        if commands[i] != commands[i + 1]:
            j = j + 1
    nodes_of_commands.append([discretized_path[-1][3], discretized_path[-1][6]])
    return nodes_of_commands
