import matplotlib.pyplot as plt
import scipy
import numpy as np


class Environment():
    objects: np.array([])

    def __init__(self, Qsize):
        self.Qsize = Qsize

        self.width = Qsize      # X-Coordinate
        self.height = Qsize     # Y-Coordinate
        self.depth = Qsize      # Z-Coordinate


    def addObject(self, pg):
        pass

    def buildEnvironment(self):
        pass

    def plotEnvironment(self):
        pass





