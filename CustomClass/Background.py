import numpy as np


class Background:
    def __init__(self, color):
        self.color = color

    def colorInPoint(self, intersectionPoint, light, camera):
        return self.color

    def getNormalVector(self, intersectionPoint):
        return np.array([0,0,0])