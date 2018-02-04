# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from matplotlib.collections import LineCollection
import time
from scipy.spatial import cKDTree
import math as ma

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
    file = open(filename, "r")
    x = []
    y = []
    for line in file:
        lista = line.rstrip().strip("{").strip("}")
        coord_ab = lista.split(",")
        x.append(float(coord_ab[1]) * np.pi / 180)
        y.append(float(np.log(np.tan((np.pi / 4) + (np.pi * float(coord_ab[0])) / 360))))
    coord_xy = np.array([x, y])
    file.close()
    return coord_xy


# Uppgift 2. och 5.

def plot_points2(coord_list, path):
    lines = []
    line = []
    for i in range(len(path) - 1):
        x0 = coord_list[0, int(path[i])]
        y0 = coord_list[1, int(path[i])]
        line.append((x0, y0))
        x1 = coord_list[0, int(path[i + 1])]
        y1 = coord_list[1, int(path[i + 1])]
        line.append((x1, y1))
        lines.append(line)
        line = []
    line_segments = LineCollection(lines, linewidths=4.5, colors='r', zorder=3)
    fig = plt.figure(1)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.add_collection(line_segments)


def plot_points(coord_list, connections, path):
    plt.plot(coord_list[0, :], coord_list[1, :], 'ob', markersize=1, zorder=2)
    lines = []
    line = []
    a = np.array((connections[0, :]))
    b = np.array((connections[1, :]))
    soize = np.size(a)
    for i in np.arange(soize):
        x0 = coord_list[0, int(a[i])]
        y0 = coord_list[1, int(a[i])]
        line.append((x0, y0))
        x1 = coord_list[0, int(b[i])]
        y1 = coord_list[1, int(b[i])]
        line.append((x1, y1))
        lines.append(line)
        line = []
    line_segments = LineCollection(lines, colors='g', zorder=1, linewidths=0.15)
    fig = plt.figure(1)
    ax = fig.gca()
    ax.add_collection(line_segments)
    plot_points2(coord_list, path)


# Uppgift 3.
def distance(x0, x1, y0, y1):
    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return dist


def construct_graph_connections(coord_list, radius):
    x_coord = coord_list[0, :]
    y_coord = coord_list[1, :]
    points = []
    real_dist = []
    for i in range(len(x_coord)):
        for z in range(1 + i, len(x_coord)):
            dist = distance(x_coord[i], x_coord[z], y_coord[i], y_coord[z])
            if dist < radius:
                points.append([i, z])
                real_dist.append(dist)
    points = np.array(points)
    points = points.transpose()
    return points, real_dist


# Uppgift 4.


def construct_graph(indices, distances):
    N = int(np.max(indices[1, :]) + 1)
    matrix = csr_matrix((distances, (indices[0, :], indices[1, :])), shape=(N, N))
    return matrix


def shortest_path(graph, start, end, direction):
    """
    :param graph: a csr graph
    :param start: start node
    :param end: end node
    :param direction: if the graph is directed or not
    :return: shortest distance and a list of predecessors
    """
    pre_list = []
    shortest, pre_all = dijkstra(graph, directed=direction, indices=[start, end], return_predecessors=True)
    i = end
    pre_list.append(end)
    while pre_all[0, i] != start:
        i = pre_all[0, i]
        pre_list.append(i)
    pre_list.append(start)
    pre_list.reverse()
    return shortest, pre_list


def construct_fast_graph_connections(coord_list, radius):
    points = np.c_[coord[0, :], coord_list[1, :]]
    tree = cKDTree(points)
    x_coord = coord_list[0, :]
    y_coord = coord_list[1, :]
    t = time.time()
    stuff = tree.query_ball_point(points, radius)
    points = []
    real_dist = []
    t = time.time()
    test = []
    for i in range(len(stuff)):
        x1 = x_coord[i]
        y1 = y_coord[i]
        test.append(i)
        #l3 = [x for x in stuff[i] if x not in test]  # Used to shorten points, makes it non-directed
        for j in stuff[i]:
            if j != i:
                x2 = x_coord[j]
                y2 = y_coord[j]
                dist = ma.sqrt((x2-x1)**2 + (y2-y1)**2)
                points.append([i, j])
                real_dist.append(dist)
    points = np.array(points)
    real_dist = np.array(real_dist)
    points = points.transpose()
    return points, real_dist


t_total = time.time()
# User interface, choose city and fast or slow construct_graph
name = "HungaryCities.txt"
radius = HUNGARY_RADIUS
start = HUNGARY_START
end = HUNGARY_END
fast_graph = True  # True if fast, False if slow


t = time.time()
coord = read_coordinate_file(name)
print("The time it takes to read_coordinate_file: ", time.time() - t)


t = time.time()
if fast_graph:
    points, dist = construct_fast_graph_connections(coord, radius)
    directed = True
else:
    points, dist = construct_graph_connections(coord, radius)
    directed = False
print("The time it takes to construct_fast_graph_connections: ", time.time() - t)

t = time.time()
csr = construct_graph(points, dist)
print("The time it takes to construct_graph: ", time.time() - t)


t = time.time()
dist_matrix, predecesor = shortest_path(csr, start, end, directed)
print("The time it takes to calculate 6 and 7: ", time.time() - t)

print("The total time excluding the plot task: ", time.time() - t_total)

t = time.time()
plot_points(coord, points, predecesor)
print("The time it takes to plot_points: ", time.time() - t)

print("The shortest distance is: ", dist_matrix[0, end])
print("The shortest path is: ", predecesor)


plt.show()

# Save the result in files
if name == "SampleCoordinates.txt":
    file_result = open("SampleCoordinatesResult.txt", "w")
    file_result.write("The shortest distance:\n")
    file_result.write(str(dist_matrix[0, end]) + "\n")
    file_result.write("The cities: \n")
    for line in predecesor:
        file_result.write(str(line) + "\n")
    file_result.close()
    print("SampleCoordinateResult.txt file written")
elif name == "HungaryCities.txt":
    file_result = open("HungaryCitiesResult.txt", "w")
    file_result.write("The shortest distance:\n")
    file_result.write(str(dist_matrix[0, end])+"\n")
    file_result.write("The cities: \n")
    for line in predecesor:
        file_result.write(str(line)+"\n")
    file_result.close()
    print("HungaryCitiesResult.txt file written")
elif name == "GermanyCities.txt":
    file_result = open("GermanyCitiesResult.txt", "w")
    file_result.write("The shortest distance:\n")
    file_result.write(str(dist_matrix[0, end]) + "\n")
    file_result.write("The cities: \n")
    for line in predecesor:
        file_result.write(str(line) + "\n")
    file_result.close()
    print("GermanyCitiesResult.txt file written")
