import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


class Polygon():
    x: np.array([])
    y: np.array([])
    # z: np.array([])
    coord: np.array([])

    def __init__(self, xPos1,  xPos2, xPos3, yPos1, yPos2, yPos3, zPos1, zPos2, zPos3):
        self.xs = np.array([xPos1,  xPos2, xPos3])
        self.ys = np.array([yPos1, yPos2, yPos3])
        self.zs = np.array([zPos1, zPos2, zPos3])

        # self.z = np.array([zPos1, zPos2, zPos3])
        self.setCoord()

    def setCoord(self):
        self.coord = np.concatenate(([self.xs], [self.ys], [self.zs]), axis=0)

        #self.coord = np.concatenate((self.coord, [self.coord[0]]), axis=0)
        print(self.coord)

        self.drawPolygon()



    def drawPolygon(self):

        verts = [list(zip(*self.coord))]
        print(verts)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.add_collection3d(Poly3DCollection(verts),zs=self.zs, zdir='y')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim3d(0,5)
        ax.set_ylim3d(0, 5)
        ax.set_zlim3d(0, 5)

        plt.show()


testPol = Polygon(0,1,1, 0,0,1, 0,0,0)
