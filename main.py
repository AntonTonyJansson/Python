#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.collections import LineCollection

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


def read_coordinate_file(filename):
    file = open(filename, "r")
    x = []
    y = []
    for line in file:           #Kan man använda np.arrays att spara värdena i istället för en list?
        lista = line.rstrip().strip("{").strip("}")
        coord_ab = lista.split(", ")
        x.append(float(coord_ab[1])*np.pi/180)
        y.append(float(np.log(np.tan((np.pi/4) + (np.pi*float(coord_ab[0]))/360))))
        #print("x/b column", x)
        #print("y/a column", y)
    coord_xy = np.array([x, y])
    return coord_xy


def plot_points(coord_name, connections):
    plt.plot(coord_name[0, :], coord_name[1, :], 'or')
    a = np.array((connections[0, :]))
    b = np.array((connections[1, :]))
    print(np.column_stack((a, b)))
    line_segments = LineCollection((np.column_stack((a, b))))
    print(line_segments)



def distance(x0, x1, y0, y1):
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    return dist

def construct_graph_connections(coord_list, radius):
    x_coord = coord_list[0, :]
    y_coord = coord_list[1, :]
    points = np.empty([0, 0])
    #print("points stil: ", points.reshape(0, 2))
    real_dist = np.empty(0)

    for i in range(len(x_coord)):
        for z in range(1+i, len(x_coord)):
            dist = distance(x_coord[i], x_coord[z], y_coord[i], y_coord[z])
            #print(dist, "Och z är = ", z, "Och i är = ", i)
            if dist < radius:
                points = np.append(points, [i, z])
                #print("Print points: ", points)
                real_dist = np.append(real_dist, dist)
    #print(points)
    points = points.reshape(int(len(points)/2), 2)      #Var försiktig med reshape!!!
    points = points.transpose()
    #print("hallå eller",points)
    real_dist = real_dist.reshape(1, int(len(real_dist)))
    #print(real_dist)
    return points, real_dist


def construct_graph(indices, distances):
    M = int(np.max(indices[0, :])+1)
    N = int(np.max(indices[1, :])+1)
    #print(N, M)
    matrix = csr_matrix((distances[0, :], (indices[0, :], indices[1, :])), shape=(M, N))
    return matrix


name = "SampleCoordinates.txt"

coord = read_coordinate_file(name)
#print(coord[0,:])

#print(coord)

#plot_points(coord)
#plt.show()

points, dist = construct_graph_connections(coord, SAMPLE_RADIUS)
#print(len(dist[0, :]))
#print((dist[0, :]))
#print(len(points[0, :]))
#print(len(points[1, :]))
#print(points)
csr = construct_graph(points, dist)
print(type(points))

plot_points(coord, points)