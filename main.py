import numpy as np
sddsadas
lite fixes...
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
    for line in file:
        print(line.rstrip())


name = "SampleCoordinates.txt"

read_coordinate_file(name)

a = np.pi
kod
