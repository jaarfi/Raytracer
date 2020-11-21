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
        # Wall 1
        #self.env.append(pg.Polygon(0,10,0, 10,10,10, 0,0,10))
        #self.env.append(pg.Polygon(10,10,0, 10,10,10, 0,10,10))

        # Wall 2
        #self.env.append(pg.Polygon(0,0,0, 0,10,10, 0,0,10))
        #self.env.append(pg.Polygon(0,0,0, 0,0,10, 0,10,10))

        self.env.append(pg.Polygon(4,8,8, 4,4,8, 0,0,0))
        self.env.append(pg.Polygon(4,8,8, 4,4,8, 6,6,6))
        self.env.append(pg.Polygon(4,8,8, 4,4,4, 0,0,6))
        self.env.append(pg.Polygon(4,4,8, 4,4,4, 0,6,6))
        self.env.append(pg.Polygon(4,4,8, 8,4,8, 6,6,6))
        self.env.append(pg.Polygon(8,8,8, 4,8,4, 0,0,6))
        self.env.append(pg.Polygon(8,8,8, 8,8,4, 6,0,6))
        self.env.append(pg.Polygon(4,4,8, 4,8,8, 0,0,0))
        self.env.append(pg.Polygon(4,4,4, 4,4,8, 0,6,0))
        self.env.append(pg.Polygon(4,4,4, 4,8,8, 6,6,0))
        self.env.append(pg.Polygon(4,8,8, 8,8,8, 0,0,6))
        self.env.append(pg.Polygon(4,4,8, 8,8,8, 0,6,6))

        self.plotEnvironment()


    def plotEnvironment(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        for i in range(len(self.env)):
            ax.add_collection3d(Poly3DCollection(self.env[i].getVert()), self.env[i].zs, zdir='y')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim3d(0, self.Qsize)
        ax.set_ylim3d(0, self.Qsize)
        ax.set_zlim3d(0, self.Qsize)


        plt.show()


env1 = Environment(10)
