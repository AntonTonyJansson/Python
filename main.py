# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from matplotlib.collections import LineCollection
import time
from scipy.spatial import cKDTree
from scipy import spatial

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


def symmetrize(a):
    return a + a.T -np.diag(a.diagonal())


def read_coordinate_file(filename):
    file = open(filename, "r")
    x = []
    y = []
    for line in file:  # Kan man använda np.arrays att spara värdena i istället för en list?
        lista = line.rstrip().strip("{").strip("}")
        coord_ab = lista.split(",")
        x.append(float(coord_ab[1]) * np.pi / 180)
        y.append(float(np.log(np.tan((np.pi / 4) + (np.pi * float(coord_ab[0])) / 360))))
        # print("x/b column", x)
        # print("y/a column", y)
    coord_xy = np.array([x, y])
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
    line_segments = LineCollection(lines, linewidths=4.5, colors='r')
    fig = plt.figure(1)
    ax = fig.gca()
    ax.add_collection(line_segments)


def plot_points(coord_list, connections, path):   # Ändra till coord_list istället för coord_name?
    plt.plot(coord_list[0, :], coord_list[1, :], 'ob', markersize=1.5)
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
    line_segments = LineCollection(lines, colors='g')
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
    points = np.empty([0, 0])
    real_dist = np.empty([0])

    for i in range(len(x_coord)):
        for z in range(1 + i, len(x_coord)):
            dist = distance(x_coord[i], x_coord[z], y_coord[i], y_coord[z])
            # print(dist, "Och z är = ", z, "Och i är = ", i)
            if dist < radius:
                points = np.append(points, [i, z])
                # print("Print points: ", points)
                real_dist = np.append(real_dist, dist)
    # print(points)
    points = points.reshape(int(len(points) / 2), 2)  # Var försiktig med reshape!!!
    points = points.transpose()
    # print("hallå eller",points)
    #print(real_dist)
    return points, real_dist


# Uppgift 4.


def construct_graph(indices, distances):
    M = int(np.max(indices[0, :]) + 1)
    N = int(np.max(indices[1, :]) + 1)
    #print(len(distances))
    matrix = csr_matrix((distances, (indices[0, :], indices[1, :])), shape=(N, N))
    return matrix


def shortest_path(graph, start, end, direction):
    pre_list = []
    shortest, pre_all = dijkstra(graph, directed=direction, indices=[start, end], return_predecessors=True)
    #print(pre_all)
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
    tree = spatial.cKDTree(points)
    print(tree)
    x_coord = coord_list[0, :]
    y_coord = coord_list[1, :]
    t = time.time()
    nearest_dist = cKDTree.query_ball_point(tree, points, radius)
    print(nearest_dist)
    stuff = tree.query_ball_point(points, radius)
    print("Tid för fast ball point",time.time() - t)
    points = np.empty([0, 0])
    real_dist = np.empty([0])
    t = time.time()

    for o in stuff:
        print(o)

    for i in range(len(stuff)):
        for j in range(len(stuff[i])):
            if i != stuff[i][j]:
                dist = distance(x_coord[i], x_coord[stuff[i][j]], y_coord[i], y_coord[stuff[i][j]])
                points = np.append(points, (i, stuff[i][j]))
                real_dist = np.append(real_dist, dist)
    print("Loopen: ", time.time()-t)
    points = points.reshape(int(len(points) / 2), 2)  # Var försiktig med reshape!!!
    points = points.transpose()
    #print(points)
    #print(real_dist)
    return points, real_dist


# CITY
name = "SampleCoordinates.txt"
radius = SAMPLE_RADIUS
start = SAMPLE_START
end = SAMPLE_END


t = time.time()
coord = read_coordinate_file(name)
print("The time it takes to read_coordinate_file: ", time.time() - t)
#print(coord)


t = time.time()
points, dist = construct_graph_connections(coord, radius)
print("The time it takes to construct_graph_connections: ", time.time() - t)
#print("Dist från 3: ", dist)
#print("Points från 3: ", points)

t = time.time()
points_fast, dist_fast = construct_fast_graph_connections(coord, radius)
print("The time it takes for fast graph: ", time.time() - t)

t = time.time()
csr = construct_graph(points_fast, dist_fast)
print("The time it takes to construct_graph: ", time.time() - t)




#print(csr)
t = time.time()
dist_matrix, predecesor = shortest_path(csr, start, end, True)
print("The time it takes to calculate shortest_path: ", time.time() - t)
#print(dist_matrix[0, end])
#print(predecesor)


t = time.time()
plot_points(coord, points, predecesor)
print("The time it takes to print: ", time.time() - t)
#plot_points2(coord, points, predecesor)
plt.show()

