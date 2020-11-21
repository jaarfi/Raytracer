import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import Polygon as pg


class Environment:

    def __init__(self, Qsize):
        self.Qsize = Qsize

        self.width = Qsize  # X-Coordinate
        self.height = Qsize  # Y-Coordinate
        self.depth = Qsize  # Z-Coordinate

        self.env = []

        self.buildEnvironmentOne()


    def buildEnvironmentOne(self):
        self.env.append(pg.Polygon(0,1,1, 0,0,1, 0,0,0))
        self.env.append(pg.Polygon(0,0,1, 1,0,1, 0,0,0))

        # Wall 1
        self.env.append(pg.Polygon(0,5,0, 5,5,5, 0,0,5))
        self.env.append(pg.Polygon(5,5,0, 5,5,5, 0,5,5))

        self.plotEnvironment()


    def plotEnvironment(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.add_collection3d(Poly3DCollection(self.env[0].getVert()), self.env[0].getZS(), zdir='y')
        ax.add_collection3d(Poly3DCollection(self.env[1].getVert()), self.env[1].getZS(), zdir='y')
        ax.add_collection3d(Poly3DCollection(self.env[2].getVert()), self.env[2].getZS(), zdir='y')
        ax.add_collection3d(Poly3DCollection(self.env[3].getVert()), self.env[3].getZS(), zdir='y')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim3d(0, 5)
        ax.set_ylim3d(0, 5)
        ax.set_zlim3d(0, 5)

        plt.show()


env1 = Environment(200)
