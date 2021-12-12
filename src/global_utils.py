import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
from motion import *


def convert_to_np(obs_coords, start, goal):
    """
    Converts the x and y coordinates of the extended obstacles, start, and goal to numpy array
    :param obs_coords: obstacles' x and y coordinates [list of inner lists of 2-element inner lists]
    :param start: start's x and y coordinates [list of x and y]
    :param goal: goal's x and y coordinates [list of x and y]
    :return: numpy version of obs_coords [list of numpy 2d arrays], start [numpy 1d array], and goal [numpy 1d array]
    """
    start = np.array(start)
    goal = np.array(goal)
    for i in range(len(obs_coords)):
        obs_coords[i] = np.array(obs_coords[i])
    return obs_coords, start, goal


def sort_obs_coords(obs_coords, clockwise=True):
    """
    Sorts the coordinates of obstacles in a clockwise or counterclockwise order w.r.t. their centers
    :param obs_coords: obstacles' x and y coordinates [list of numpy 2d arrays]
    :param clockwise: [boolean] | True (sort clockwise), False (sort counterclockwise)
    :return: sorted version of obs_coords  [list of numpy 2d arrays]
    """
    for i in range(len(obs_coords)):
        obs = obs_coords[i]
        # Finding the center coordinates of the obstacle
        cx = np.mean(obs[:, 0])
        cy = np.mean(obs[:, 1])
        # Calculating the orientation (angle) of a line segment in between the obstacle's center and each vertex
        # of the obstacle
        angles = [np.arctan2(x - cx, y - cy) for (x, y) in obs]
        # Finding the index of each angle while sorting the 'angles' in ascending order
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
    :param obs_coords: obstacles' x and y coordinates [list of numpy 2d arrays]
    :param start: start's x and y coordinates [numpy 1d array]
    :param goal: goal's coordinates [numpy 1d array]
    :param edges: edges of the visibility graph [list of lists]
    :param optimal_path: optimal path derived by A* [list of lists]
    :return: obs_coords_swap [list of numpy 2d arrays],
             start_swap [numpy 1d array], goal_swap [numpy 1d array],
             edges_swap (only the coordinates of vertices) [list of lists],
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
    Plots the environment, the visibility graph, and the optimal path
    :param obs_coords: swapped obstacles' x and y coordinates [list of numpy 2d arrays]
    :param start: swapped start's x and y coordinates [numpy 1d array]
    :param goal: swapped goal's x and y coordinates [numpy 1d array]
    :param edges: swapped x and y coordinates of the vertices of the edges [list of lists]
    :param optimal_path: swapped x and y coordinates of the vertices of the optimal path [list of lists]
    :return: None
    """
    # Map size
    x_size = 720
    y_size = 1280
    # PLot ptions
    x_step = 80
    y_step = 160
    plt.figure(1)
    plt.grid(True)
    plt.axis('square')
    plt.axis([0, y_size, 0, x_size])
    plt.xticks(np.arange(0, y_size + y_step, step=y_step))
    plt.yticks(np.arange(0, x_size + x_step, step=x_step))
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
            plt.plot([obs[i, 0], obs[i + 1, 0]], [obs[i, 1], obs[i + 1, 1]], '-', color='#0072BD',
                     marker='.', markerfacecolor='#D95319', markeredgecolor='#D95319', markersize=10)
            plt.text(obs[i, 0], obs[i, 1], str(count))
            count = count + 1
    # Edges
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k--', linewidth=0.5)
    # Optimal Path
    for i in range(len(optimal_path) - 1):
        plt.plot([optimal_path[i][0], optimal_path[i + 1][0]], [optimal_path[i][1], optimal_path[i + 1][1]],
                 '-', color='#77AC30')
    plt.text(start[0], start[1], '1')
    plt.text(goal[0], goal[1], str(count))


def print_data(*argv):
    """
    Print the desired variables
    :param: arbitrary number of variables
    :return: None
    """
    for arg in argv:
        print(len(arg))
        for item in arg:
            print(item)
        print('------------------------------------------------------------')


def create_obs_shape(obs_coords):
    """
    Creates polygons from the coordinates of the obstacles using Shapely module
    :param obs_coords: obstacles' x and y coordinates [list of numpy 2d arrays]
    :return: obs_shapes [list of shapely.geometry.polygon.Polygon objects]
    """
    obs_shapes = []
    for obs in obs_coords:
        obs_shapes.append(sg.Polygon(obs))
    return obs_shapes


def is_node_reachable(node, map_margin=30, x_size=720, y_size=1280):
    """
    Checks whether a node is within our desired environment or not
    :param node: a node of the visibility graph [list of 4 elements]
    :param map_margin: the internal margin (in mm) of the environment [an integer]
    :param x_size: the length (in mm) of the x-axis in our desired reference system [an integer]
    :param y_size: the length (in mm) of the y-axis in our desired reference system [an integer]
    :return: [boolean] | True (reachable), False (not reachable)
    """
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
    """
    Creates the visibility graph containing a set of nodes and edges in between them
    :param obs_coords: obstacles' x and y coordinates [list of numpy 2d arrays]
    :param start: start's x and y coordinates [numpy 1d array]
    :param goal: goal's x and y coordinates [numpy 1d array]
    :return: nodes [list of 4-element inner lists], edges [list of 5-element inner lists]
    """
    # Number of nodes = 2 (start + goal) + number of the vertices of the obstacles
    num_nodes = 2
    for obs in obs_coords:
        num_nodes = num_nodes + obs.shape[0]
    # Generating nodes
    # format: [node's label (1 for start, total number of nodes for goal),\
    #          obstacle's number (0 for start, len(obs_coords)+1 for goal),\
    #          numpy 1d array of x and y coordinates,\
    #          the number of vertices of the obstacle to which the node belong (0 for start and goal)]
    nodes = [[1, 0, start, 0]]
    j = 2
    for i in range(len(obs_coords)):
        obs = obs_coords[i]
        for vrtc in obs:
            nodes.append([j, i + 1, vrtc, len(obs)])
            j = j + 1
    nodes.append([num_nodes, len(obs_coords) + 1, goal, 0])
    # Generating edges
    # format: [1st node's label, 1st node's x and y coordinates,\
    #          2nd node's label, 2nd node's x and y coordinates,\
    #          length of the edge]
    edges = []
    for i in range(num_nodes):
        # Checking whether the chosen node is reachable or not
        if not is_node_reachable(nodes[i]):
            continue
        for j in range(i + 1, num_nodes):
            if not is_node_reachable(nodes[j]):
                # Checking whether the chosen node is reachable or not
                continue
            elif (j - i > 1) and (nodes[j][1] == nodes[i][1]):
                # Skipping the obstacles' diagonals
                continue
            elif (j - i == 1) and (nodes[j][1] - nodes[i][1] == 0):
                # Adding the obstacles' boundaries
                edges.append([nodes[i][0], nodes[i][2], nodes[j][0], nodes[j][2],\
                              np.linalg.norm(nodes[i][2] - nodes[j][2])])
            else:
                if (j - i == 1) and (nodes[j][1] - nodes[i][1] == 1) and (nodes[i][1] != 0):
                    # Adding the last boundary edge of obstacles
                    temp = i - nodes[i][3] + 1
                    edges.append([nodes[temp][0], nodes[temp][2], nodes[i][0], nodes[i][2],\
                                  np.linalg.norm(nodes[temp][2] - nodes[i][2])])
                # Creating a line segment (edge) between 1st and 2nd node using Shapely module
                line = sg.LineString(np.vstack((nodes[j][2], nodes[i][2])))
                # Creating polygons from the coordinates of the obstacles using Shapely module
                obs_shapes = create_obs_shape(obs_coords)
                # Checking whether the edge contains two vertices of any boundary edges of obstacles
                # so that we can realize whether an edge is collinear with any boundary edges of obstacles
                flag = 0
                for obs in obs_shapes:
                    check1 = [line.contains(sg.Point(coord)) for coord in obs.boundary.coords]
                    if any(check1):
                        flag = 1
                        break
                if flag == 1:
                    # Skipping the collinear edges
                    continue
                # Checking whether the edge cross any obstacle or not
                check2 = [line.crosses(obs) for obs in obs_shapes]
                if not any(check2):
                    # Adding the edges which are not crossing any obstacles
                    edges.append([nodes[i][0], nodes[i][2], nodes[j][0], nodes[j][2],\
                                  np.linalg.norm(nodes[i][2] - nodes[j][2])])
    # Sorting the edges according to the labels of the 1st nodes and then according to the length of the edge
    edges.sort(key=lambda x: (x[0], x[4]))
    return nodes, edges


def heuristic(nodes):
    """
    Calculating the heuristic (h) for A* algorithm using the Euclidean distance of each node to the goal
    :param nodes: [list of 4-element inner lists]
    :return: h [list of 2-element inner lists]
    """
    goal = nodes[-1][2]
    h = []
    # format: [node's label, node's Euclidean distance to the goal]
    # The 1st element (node's label) is useful for debugging and makes the result more readable.
    for i in range(len(nodes)):
        h.append([nodes[i][0], np.linalg.norm(goal - nodes[i][2])])
    return h


def generate_path(path_so_far, current, nodes):
    """
    Generates the optimal path from the start node to the current node (goal)
    :param path_so_far: generated path by A* algorithm containing the immediately preceding node for each particular
                        node based on the cheapest path from start to that particular node [a dictionary of node labels]
    :param current: current node label (goal)
    :param nodes: [list of 3-element inner lists]
    :return: optimal_path [list of 2-element inner lists contains the labels and coordinates of the nodes]
    """
    optimal_path = [[current, nodes[current - 1][2]]]
    while current in path_so_far.keys():
        # Adding where the current node came from to the start of the list
        optimal_path.insert(0, [path_so_far[current], nodes[path_so_far[current] - 1][2]])
        current = path_so_far[current]
    return optimal_path


def a_star(nodes, edges, h):
    """
    Implements A* algorithm to find the optimal path from start to goal
    :param nodes: nodes of the visibility graph [list of 4-element inner lists]
    :param edges: edges of the visibility graph [list of 5-element inner lists]
    :param h: heuristic [list of 2-element inner lists]
    :return: calls the 'generate_path' function to return the optimal_path or
             raise exceptions or
    """
    # Start label
    start = nodes[0][0]
    # Goal label
    goal = nodes[-1][0]
    # Past cost (g score): the cost of the cheapest path from start to each node
    # format: {label of the node: cost}
    past_cost = {start: 0}
    # Total cost (f score): f(n) = g(n) + h(n)
    # format: {label of the node: cost}
    total_cost = {start: past_cost[start] + h[0][1]}
    # The set of visited nodes for which the neighbors need to be explored
    # format: [label of the nodes]
    open_set = [start]
    # The set of visited nodes for which the neighbors no longer need to be explored
    closed_set = []
    # The path that is going to be generated by A* algorithm containing the immediately preceding node for each
    # particular node based on the cheapest path from start to that particular node
    # format: {label of a particular node: label of the node immediately preceding it that particular node based on the
    #          cheapest path from start to that particular node}
    path_so_far = {}
    # Initializing the past_cost and total_cost (except for the start node)
    for i in range(1, len(nodes)):
        past_cost[nodes[i][0]] = np.inf
        total_cost[nodes[i][0]] = np.inf
    # Implementing A*
    while open_set != []:
        # Choosing the node in 'open_set' which has the lowest total_cost
        total_cost_open = {key: val for (key, val) in total_cost.items() if key in open_set}
        current = min(total_cost_open, key=total_cost_open.get)
        # Checking whether the current node (label) is equal to the goal (label) or not
        if current == goal:
            # Calling the 'generate_path' function to return the optimal_path
            return generate_path(path_so_far, current, nodes)
        open_set.remove(current)
        closed_set.append(current)
        # Calculating the neighbors of the current node
        # format: [label of the neighbor, the length of the edge between the current node and its neighbor]
        # It is noteworthy that sometimes the current node is the 1st node's label and sometimes the 2nd node's label
        # of an edge
        neighbors1 = [[edge[2], edge[4]] for edge in edges if edge[0] == current]
        neighbors2 = [[edge[0], edge[4]] for edge in edges if edge[2] == current]
        neighbors = neighbors1 + neighbors2
        neighbors.sort(key=lambda x: (x[0], x[1]))
        for neighbor in neighbors:
            # Checking if the neighbor is already explored
            if neighbor[0] in closed_set:
                continue
            # Tentative past cost (the distance from start to the neighbor node passing through the current node)
            tentative_past_cost = past_cost[current] + neighbor[1]
            if tentative_past_cost < past_cost[neighbor[0]]:
                # Recording The path to the neighbor which is cheaper than any of the previous ones based on
                # the tentative past cost
                path_so_far[neighbor[0]] = current
                past_cost[neighbor[0]] = tentative_past_cost
                total_cost[neighbor[0]] = past_cost[neighbor[0]] + h[neighbor[0] - 1][1]
                if neighbor[0] not in open_set:
                    open_set.append(neighbor[0])
    raise Exception('Failed to find a path to the goal!')


def calc_angle(p1, p2):
    """
    Calculates the orientation (angle) of a line connecting two points in our desired reference system
    :param p1: the x and y coordinates of the 1st point [numpy 1d array]
    :param p2: the x and y coordinates of the 2nd point [numpy 1d array]
    :return: angle [a float in radian]
    """
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    # Making sure the angle is always between 0 and 2*pi radians
    if angle < 0:
        angle = angle + 2 * np.pi
    return angle


def discretize_path(start_po, edges, optimal_path):
    """
    Discretizes the optimal path to a set of rotations and angles
    :param start_po: start's x, y, and theta coordinates [list of x, y, th]
    :param edges: edges of the visibility graph [list of 5-element inner lists]
    :param optimal_path: the optimal path
                         [list of 2-element inner lists contains the labels and coordinates of the nodes]
    :return: discretized_path [list of 8-element inner lists]
    """
    # Storing the amount of rotations and linear movements between the nodes of optimal path
    movements = []
    for i in range(len(optimal_path) - 1):
        nodei0 = optimal_path[i]
        nodei1 = optimal_path[i + 1]
        angle = calc_angle(nodei0[1], nodei1[1])
        movements.append(angle)
        dist = [edge[4] for edge in edges if ((edge[0] == nodei0[0]) and (edge[2] == nodei1[0]))
                or ((edge[2] == nodei0[0]) and (edge[0] == nodei1[0]))][0]
        movements.append(dist)
    discretized_path = []
    # format [1 (rotation) | 2 (linear movement), amount of rotation or liner movement,
    #         1st node's label, 1st node's x and y coordinates, 1st node's orientation,
    #         2nd node's label, 2nd node's x and y coordinates, 2nd node's orientation]
    discretized_path.append([1, movements[0] - start_po[-1], 1, np.array(start_po[0:-1]), start_po[-1],\
                             optimal_path[1][0], optimal_path[1][1], movements[0]])
    for i in range(1, len(movements)):
        idx = i // 2
        if i % 2 == 1:
            discretized_path.append([2, movements[i], optimal_path[idx][0], optimal_path[idx][1], movements[i - 1],\
                                     optimal_path[idx + 1][0], optimal_path[idx + 1][1], movements[i - 1]])
        else:
            discretized_path.append([1, movements[i] - movements[i - 2], optimal_path[idx][0], optimal_path[idx][1],\
                                     movements[i - 2], optimal_path[idx + 1][0], optimal_path[idx + 1][1],
                                     movements[i]])
    return discretized_path


def do_global_navigation(obs_coords, start_po, goal_po):
    """
    Does all the global navigation part and returns the optimal and discretized optimal path
    :param obs_coords: obstacles' x and y coordinates [list of lists of 2-element lists]
    :param start_po: start's x, y, and theta coordinates [list of x, y, th]
    :param goal_po: goal's x, y, and theta coordinates [list of x, y, th]
    :return: optimal_path [list of 2-element inner lists contains the labels and coordinates of the nodes],
             discretized_path [list of 8-element inner lists],
    """
    [obs_coords, start, goal] = convert_to_np(obs_coords, start_po[0:-1], goal_po[0:-1])
    obs_coords = sort_obs_coords(obs_coords)
    [nodes, edges] = vis_graph(obs_coords, start, goal)
    h = heuristic(nodes)
    optimal_path = a_star(nodes, edges, h)
    discretized_path = discretize_path(start_po, edges, optimal_path)
    return discretized_path, optimal_path


def test_global_navigation(obs_coords, start_po, goal_po):
    """
    Tests all the global navigation part and returns all the required data
    :param obs_coords: obstacles' x and y coordinates [list of lists of 2-element lists]
    :param start_po: start's x, y, and theta coordinates [list of x, y, th]
    :param goal_po: goal's x, y, and theta coordinates [list of x, y, th]
    :return: nodes [list of 4-element inner lists], edges [list of 5-element inner lists],
             optimal_path [list of 2-element inner lists contains the labels and coordinates of the nodes],
             discretized_path [list of 8-element inner lists],
    """
    [obs_coords, start, goal] = convert_to_np(obs_coords, start_po[0:-1], goal_po[0:-1])
    obs_coords = sort_obs_coords(obs_coords)
    [nodes, edges] = vis_graph(obs_coords, start, goal)
    h = heuristic(nodes)
    optimal_path = a_star(nodes, edges, h)
    discretized_path = discretize_path(start_po, edges, optimal_path)
    return obs_coords, start, goal, nodes, edges, optimal_path, discretized_path


def calibrate_path(current_po, node):
    """
    Gives the angle (in radians) and the distance (in mm) from the line that connects thymio to the next node
    :param current_po: current x, y, and theta coordinates of thymio [numpy 1d array]
    :param node: the x and y coordinates of the next node [numpy 1d array]
    :return: angle [a float], distance [a float]
    """
    angle = calc_angle(current_po[0:-1], node)
    angle = angle - current_po[-1]
    dist = np.linalg.norm(node - current_po[0:-1])
    return angle, dist


def get_commands(thym_state, val):
    """
    Calls one of the move_forward, move_backward, turn_left, turn_right functions based on the thymio's state
    :param thym_state: [an integer] | 1 (rotation), 2 (linear movement)
    :param val: the desired amount of linear movement or rotation [a float]
    :return: calls one of the functions mentioned above to return a set of motor commands
             [list of 2-element inner lists]
    """
    if thym_state == 2 and val > 0:       # Forward
        return move_forward(val)
    elif thym_state == 2 and val < 0:     # Backward
        return move_backward(val)
    elif thym_state == 1 and val < 0:     # Rotate Right
        return turn_right(val)
    elif thym_state == 1 and val > 0:     # Rotate Left
        return turn_left(val)


def calc_commands(discretized_path):
    """
    Generates all the required motor commands which make thymio move from start to the goal
    :param discretized_path: the discretized optimal path [list of 8-element inner lists]
    :return: commands [list of 2-element inner lists]
    """
    commands = []
    for item in discretized_path:
        commands.extend(get_commands(item[0], item[1]))
    return commands
