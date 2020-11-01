import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np



class Main():
    def __init__(self, aspect_ratio, width):
        self.window_width = width
        self.aspect_ratio = aspect_ratio
        self.window_height = int(self.window_width * self.aspect_ratio)
        self.picture = Picture(self.window_width, self.window_height)

        self.calculatePicture()
        self.drawPicture()

    def calculatePicture(self):
        for i in range(1, self.window_width):
            subimg = []
            for j in range(1, self.window_height):
                pixel_color = [0,int(255 * i / self.window_width), 255]
                self.picture.insertPixel(pixel_color, i, j)


    def drawPicture(self):
        plt.imshow(self.picture.values)
        plt.show()

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Sphere():
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

class Picture():
    def __init__(self, length, height):
        self.values = np.array(np.zeros((length,height,3), dtype=int))
        print(self.values)
        self.length = length
        self.height = height

    def insertPixel(self, color, x, y):
        self.values[x][y] = color

main = Main(16/9, 255)