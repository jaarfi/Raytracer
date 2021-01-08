import numpy as np


class IntersecPointInformations:
    def __init__(self, distances, points, normalVectors, colors, bodies, types):
        self.distances = np.array(distances)
        self.points = np.array(points)
        self.normalVectors = np.array(normalVectors)
        #print("normals", self.normalVectors)
        self.displacedPoints = np.add(self.points, self.normalVectors * 0.01)
        self.colors = colors
        self.bodies = bodies
        self.types = types