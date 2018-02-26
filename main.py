# Imports
import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from matplotlib.collections import LineCollection
import time
from scipy.spatial import cKDTree

# CONSTANTS
SAMPLE_RADIUS = 0.08
SAMPLE_START = 0
SAMPLE_END = 5

HUNGARY_RADIUS = 0.005
HUNGARY_START = 311
HUNGARY_END = 702

GERMANY_RADIUS = 0.0025
GERMANY_START = 1573
GERMANY_END = 10584


# Uppgift 1.

np.set_printoptions(precision=4)


def read_coordinate_file(filename):
    """
    :param filename: Textfile with coordinates given in latitude and longitude
    :return: np.array with x and y coordinates
    """
    file = open(filename, "r")
    x = []
    y = []
    for line in file:
        lista = line.rstrip().strip("{").strip("}")
        coord_ab = lista.split(",")
        x.append(float(coord_ab[1]) * np.pi / 180)  # x and y coordinates calculated using Mercator projection
        y.append(float(np.log(np.tan((np.pi / 4) + (np.pi * float(coord_ab[0])) / 360))))
    coord_xy = np.array([x, y])
    file.close()
    return coord_xy


# Uppgift 2. och 5.

def plot_points2(coord_list, path):
    """
    Plotting the shortest path between two cities.
    """
    lines = []
    line = []
    for i in range(len(path) - 1):
        x0 = coord_list[0, path[i]]
        y0 = coord_list[1, path[i]]
        line.append((x0, y0))
        x1 = coord_list[0, path[i + 1]]
        y1 = coord_list[1, path[i + 1]]
        line.append((x1, y1))
        lines.append(line)
        line = []
    line_segments = LineCollection(lines, linewidths=3, colors='r', zorder=3)
    fig = plt.figure(1)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.add_collection(line_segments)


def plot_points(coord_list, connections, path):
    """
    Plotting the points and lines between the cities.
    """
    plt.plot(coord_list[0, :], coord_list[1, :], 'ob', markersize=0.2)
    lines = []
    line = []
    a = np.array((connections[0, :]))
    b = np.array((connections[1, :]))
    for i in np.arange(a.size):
        x0 = coord_list[0, a[i]]
        y0 = coord_list[1, a[i]]
        line.append((x0, y0))
        x1 = coord_list[0, b[i]]
        y1 = coord_list[1, b[i]]
        line.append((x1, y1))
        lines.append(line)
        line = []
    line_segments = LineCollection(lines, colors='g', zorder=1, linewidths=0.2)
    fig = plt.figure(1)
    ax = fig.gca()
    ax.add_collection(line_segments)
    plot_points2(coord_list, path)


# Uppgift 3.
def distance(x0, x1, y0, y1):
    dist = m.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return dist


def construct_graph_connections(coord_list, radius):
    x_coord = coord_list[0, :]
    y_coord = coord_list[1, :]
    points = []
    real_dist = []

    for i, j in enumerate(x_coord):
        for k, l in enumerate(x_coord):
            dist = distance(j, l, y_coord[i], y_coord[k])
            if dist < radius and j != l and y_coord[i] != y_coord[k]:
                points.append([i, k])
                real_dist.append(dist)
    points = np.array(points)
    points = points.transpose()
    return points, real_dist


# Uppgift 4.


def construct_graph(indices, distances, N):
    matrix = csr_matrix((distances, (indices[0, :], indices[1, :])), shape=(N, N))
    return matrix


def shortest_path(graph, start, end):
    """
    :param graph: a csr graph
    :param start: start node
    :param end: end node
    :return: shortest distance and a list of predecessors

    shortest is a list which describes the distance to all cities from the start node. Noted with the index.
    ex: distance from start to node 10 = shortest[10].

    pre_all is a predecessor list that describes which city is closest to current city.
    ex: if pre_all = [-9999 2 0 4 0 3 4], then the closest city to city 5 is pre_all[5], which is city 3.
    The closest to city 3 is then pre_all[3] = 4 and so on. -9999 means that no path exists.

    """
    pre_list = []
    # indices indicates which nodes to start from
    # return_predecessors indicates that dijkstra returns a predecessor list as well
    shortest, pre_all = dijkstra(graph, indices=start, return_predecessors=True)
    i = end
    pre_list.append(end)
    while pre_all[i] != start:  # Loop to get the predecessor list from node start to end
        i = pre_all[i]
        pre_list.append(i)
    pre_list.append(start)
    pre_list.reverse()
    return shortest, pre_list


def construct_fast_graph_connections(coord_list, radius):
    """
    :param coord_list: 2xn array of coordinates x and y for n points
    :param radius: maximum radius between reachable cities
    :return:
    points: a 2xm np.array of city combinations within the given radius. m number of combinations
        ex: points = [[0 1 ...]
                      [1 2 ...]] where city 1 can be reached from city 0 and 2 from 1.
    real_dist: 1xm np.array of distances between the reachable cities.
        ex: real_dist = [0.5 1 0.3...] where the distance between 0 and 1 is 0.5
    """
    points = coord_list.transpose()
    tree = cKDTree(points)
    x_coord = coord_list[0, :]
    y_coord = coord_list[1, :]
    reach = tree.query_ball_point(points, radius)   # Creates a list of lists where each list represent which cities can
                                                    # be reached from city index in reach
    points = []
    real_dist = []
    for i in range(len(reach)):  # Loop through all lists in reach, to determine distances between each accessible city
        x1 = x_coord[i]
        y1 = y_coord[i]
        for j in reach[i]:
            if j != i:
                x2 = x_coord[j]
                y2 = y_coord[j]
                dist = distance(x1, x2, y1, y2)
                points.append([i, j])
                real_dist.append(dist)
    points = np.array(points)
    real_dist = np.array(real_dist)
    points = points.transpose()
    return points, real_dist


t_total = time.time()
# User interface, choose city and fast or slow construct_graph
name = "SampleCoordinates.txt"
radius = SAMPLE_RADIUS
start = SAMPLE_START
end = SAMPLE_END
which_graph = True  # True if fast, False if slow


t = time.time()
coord = read_coordinate_file(name)
print("The time it takes to read_coordinate_file: ", time.time() - t)


t = time.time()
if which_graph:
    points, dist = construct_fast_graph_connections(coord, radius)
else:
    points, dist = construct_graph_connections(coord, radius)
print("The time it takes to construct_fast_graph_connections: ", time.time() - t)

N = coord[0, :].size   # Number of unique cities


t = time.time()
csr = construct_graph(points, dist, N)
print("The time it takes to construct_graph: ", time.time() - t)


t = time.time()
dist_matrix, predecesor = shortest_path(csr, start, end)
print("The time it takes to calculate 6 and 7: ", time.time() - t)

print("The total time excluding the plot task: ", time.time() - t_total)

t = time.time()
plot_points(coord, points, predecesor)
print("The time it takes to plot_points: ", time.time() - t)

print("The shortest distance is: ", dist_matrix[end])
print("The shortest path is: ", predecesor)


plt.show()

# Save the result in files
if name == "SampleCoordinates.txt":
    file_result = open("SampleCoordinatesResult.txt", "w")
    file_result.write("The shortest distance:\n")
    file_result.write(str(dist_matrix[end]) + "\n")
    file_result.write("The cities: \n")
    for line in predecesor:
        file_result.write(str(line) + "\n")
    file_result.close()
    print("SampleCoordinateResult.txt file written")
elif name == "HungaryCities.txt":
    file_result = open("HungaryCitiesResult.txt", "w")
    file_result.write("The shortest distance:\n")
    file_result.write(str(dist_matrix[end])+"\n")
    file_result.write("The cities: \n")
    for line in predecesor:
        file_result.write(str(line)+"\n")
    file_result.close()
    print("HungaryCitiesResult.txt file written")
elif name == "GermanyCities.txt":
    file_result = open("GermanyCitiesResult.txt", "w")
    file_result.write("The shortest distance:\n")
    file_result.write(str(dist_matrix[end]) + "\n")
    file_result.write("The cities: \n")
    for line in predecesor:
        file_result.write(str(line) + "\n")
    file_result.close()
    print("GermanyCitiesResult.txt file written")
