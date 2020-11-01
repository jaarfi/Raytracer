import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time




class Point(typing.NamedTuple):
    x: float
    y: float
    z: float

class Vector(typing.NamedTuple):
    x: float
    y: float
    z: float

class VectorFunctions(object):
    @staticmethod
    def subtract(vector1: Vector, vector2: Vector):
        return Vector(vector1.x-vector2.x,vector1.y-vector2.y,vector1.z-vector2.z)

    @staticmethod
    def add(vector1: Vector, vector2: Vector):
        return Vector(vector1.x+vector2.x,vector1.y+vector2.y,vector1.z+vector2.z)

    @staticmethod
    def fromTo(vector1: Vector, vector2: Vector):
        return VectorFunctions.subtract(vector2, vector1)

    @staticmethod
    def dotProduct(vector1: Vector, vector2: Vector):           #SkalarProdukt
        return vector1.x*vector2.x + vector1.y*vector2.y + vector1.z*vector2.z

class Color(typing.NamedTuple):
    r: int = 0
    g: int = 0
    b: int = 0

class Sphere(typing.NamedTuple):
    r: float
    center: Point
    color: Color

class ViewingScreen(typing.NamedTuple):
    bottom_left: tuple
    width: int
    height: int
    resolution_x: int
    resolution_y: int

class Camera(typing.NamedTuple):
    center: Point
    viewingScreen: ViewingScreen

class Pixel(typing.NamedTuple):
    point: Point

    def color(self, camera: Camera, sphere: Sphere):
        vectorToCamera = VectorFunctions.fromTo(self.point, camera.center)
        vectorToSphereCenter = VectorFunctions.fromTo(self.point, sphere.center)

        dotProd1 = VectorFunctions.dotProduct(vectorToCamera, vectorToCamera)
        dotProd2 = VectorFunctions.dotProduct(vectorToCamera, vectorToSphereCenter)
        dotProd3 = VectorFunctions.dotProduct(vectorToSphereCenter, vectorToSphereCenter)

        poly = np.poly1d([dotProd1, 2 * dotProd2, dotProd3 - sphere.r * sphere.r])
        results = poly.r

        if (not np.iscomplex(results[0]) or not np.iscomplex(results[1])):
            pixel_color = sphere.color  # rot
        else:
            pixel_color = Color(0, 123, 255)  # Normale Farbe

        return pixel_color

class Main():
    def __init__(self, sphere: Sphere, camera: Camera):
        self.screen = camera.viewingScreen
        self.window_width = self.screen.resolution_x
        self.window_height = self.screen.resolution_y
        self.picture = Picture(self.window_width, self.window_height)

        self.calculatePicture()
        self.drawPicture()

    def calculatePicture(self):
        now = time.time()
        for i in range(1, self.window_width):
            for j in range(1, self.window_height):
                currentPixelPoint = Point(*VectorFunctions.add(self.screen.bottom_left,Vector(self.screen.width/self.window_width*i,self.screen.height/self.window_height*j,0)))
                currentPixel = Pixel(currentPixelPoint)


                self.picture.insertPixel(currentPixel.color(camera,sphere), i, j)

        print(time.time()-now)


    def drawPicture(self):
        plt.imshow(self.picture.values)
        plt.show()


class Picture():
    def __init__(self, width, height):
        self.values = np.array(np.zeros((width,height,3), dtype=int))
        self.width = width
        self.height = height

    def insertPixel(self, color: Color, x, y):
        self.values[x][y] = color

camera = Camera(Point(0,0,0),ViewingScreen(Point(-2,-1,-4),4,4,200,200))
sphere = Sphere(0.4, Point(0,0,-2), Color(0,255,0))
main = Main(sphere, camera)
v = Vector(0,0,0)
