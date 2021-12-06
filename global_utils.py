import matplotlib.pyplot as plt
import numpy as np
import math
import shapely.geometry as sg

start_po = (350.75, 69.25, np.deg2rad(99.4089582505319))
goal_po = (637.25, 941.25, np.deg2rad(90))
obs_coords = [[[538, 460], [551, 239], [693, 245], [685, 461]],
              [[382, 596], [512, 527], [592, 527], [619, 661], [492, 727], [412, 727]],
              [[129, 1153], [143, 968], [354, 978], [345, 1162]],
              [[0, 468], [181, 459], [189, 643], [0, 650]]]


def convert_to_np(obs_coords, start, goal):
    start = np.squeeze(np.array(start))
    goal = np.squeeze(np.array(goal))
    for i in range(len(obs_coords)):
        obs_coords[i] = np.array(obs_coords[i])
    return obs_coords, start, goal


def sort_coords(obs_coords, clockwise=True):
    for i in range(len(obs_coords)):
        obs = obs_coords[i]
        cx = np.mean(obs[:, 0])
        cy = np.mean(obs[:, 1])
        angles = [math.atan2(x-cx, y-cy) for (x, y) in obs]
        indices = sorted(range(len(angles)), key=angles.__getitem__)
        if clockwise:
            obs = [obs[idx] for idx in indices]
            obs_coords[i] = np.array(obs)
        else:
            obs = [obs[idx] for idx in indices[::-1]]
            obs_coords[i] = np.array(obs)
    return obs_coords


def swap_xy(obs_coords, start, goal, edges, optimal_path):
    start_swap = np.copy(start)
    start_swap[1], start_swap[0] = start_swap[0], start_swap[1]
    goal_swap = np.copy(goal)
    goal_swap[1], goal_swap[0] = goal_swap[0], goal_swap[1]
    obs_coords_swap = []
    for i in range(len(obs_coords)):
        obs_coords_swap.append(np.copy(obs_coords[i]))
        obs_coords_swap[i][:, [1, 0]] = obs_coords_swap[i][:, [0, 1]]
    edges_swap = []
    for i in range(len(edges)):
        temp1 = np.copy(edges[i][1])
        temp1[1], temp1[0] = temp1[0], temp1[1]
        temp2 = np.copy(edges[i][3])
        temp2[1], temp2[0] = temp2[0], temp2[1]
        edges_swap.append([temp1, temp2])
    optimal_path_swap = []
    for i in range(len(optimal_path)):
        temp = np.copy(optimal_path[i][1])
        temp[1], temp[0] = temp[0], temp[1]
        optimal_path_swap.append(temp)
    return obs_coords_swap, start_swap, goal_swap, edges_swap, optimal_path_swap


def plot_map(obs_coords, start, goal, edges, optimal_path,warpedimg_clean):
    # Options
    x_size = 720
    y_size = 1280
    x_step = 80
    y_step = 160
    fig = plt.figure(1)
    plt.imshow(warpedimg_clean)
    # plt.grid(True)
    # plt.axis('square')
    # plt.axis([0, y_size, 0, x_size])
    # plt.xticks(np.arange(0, y_size+y_step, step=y_step))
    # plt.yticks(np.arange(0, x_size+x_step, step=x_step))
    plt.setp(plt.gca(), 'ylim', reversed(plt.getp(plt.gca(), 'ylim')))
    # plt.gca().spines['bottom'].set_position('zero')
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
    # for edge in edges:
    #     plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k--', linewidth=0.5)
    # Optimal Path
    for i in range(len(optimal_path)-1):
        plt.plot([optimal_path[i][0], optimal_path[i+1][0]], [optimal_path[i][1], optimal_path[i+1][1]],
                 '-', color='#77AC30')
    plt.text(start[0], start[1], '1-Start')
    plt.text(goal[0], goal[1], str(count)+'-Target')
    #plt.savefig('02.png')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


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
        for j in range(i+1, num_nodes):
            if (j - i > 1) and (nodes[j][1] == nodes[i][1]):    # Skipping Diagonals
                pass
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
        if current == goal:
            return generate_path(path_so_far, current, nodes)
        open_set.remove(current)
        closed_set.append(current)
        neighbors = [[edge[2], edge[4]] for edge in edges if edge[0] == current]
        for neighbor in neighbors:
            if neighbor[0] in closed_set:
                continue
            if neighbor[0] not in open_set:
                open_set.append(neighbor[0])
            tenative_past_cost = past_cost[current] + neighbor[1]
            if tenative_past_cost < past_cost[neighbor[0]]:
                path_so_far[neighbor[0]] = current
                past_cost[neighbor[0]] = tenative_past_cost
                total_cost[neighbor[0]] = past_cost[neighbor[0]] + h[neighbor[0]-1][1]
    print('Failed to find a path to the goal!')
    return []


def calc_angle(p1, p2):
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    if angle < 0:
        angle = angle + 2 * np.pi
    return angle


def discretize_path(start_po, goal_po, edges, optimal_path):
    movements = []
    for i in range(len(optimal_path)-1):
        nodei0 = optimal_path[i]
        nodei1 = optimal_path[i+1]
        angle = calc_angle(nodei0[1], nodei1[1])
        movements.append(angle)
        dist = [edge[4] for edge in edges if (edge[0] == nodei0[0]) and (edge[2] == nodei1[0])][0]
        movements.append(dist)
    discretized_path = []
    discretized_path.append([1, movements[0] - start_po[-1], 1, np.array(start_po[0:-1]), start_po[-1],\
                             optimal_path[0][0], optimal_path[0][1], movements[0]])
    for i in range(1, len(movements)):
        idx = i // 2
        print(i)
        if i % 2 == 1:
            discretized_path.append([2, movements[i], optimal_path[idx][0], optimal_path[idx][1], movements[i-1],\
                                     optimal_path[idx+1][0], optimal_path[idx+1][1], movements[i-1]])
        else:
            discretized_path.append([1, movements[i] - movements[i-2], optimal_path[idx][0], optimal_path[idx][1],\
                                     movements[i-2], optimal_path[idx+1][0], optimal_path[idx+1][1], movements[i]])
    discretized_path.append([1, goal_po[-1] - movements[-2], optimal_path[idx+1][0], optimal_path[idx+1][1],\
                             movements[i-1], optimal_path[idx+1][0], np.array(goal_po[0:-1]), goal_po[-1]])
    return discretized_path

