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
        return Vector(vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z)

    @staticmethod
    def add(vector1: Vector, vector2: Vector):
        return Vector(vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z)

    @staticmethod
    def addScalarToAll(vector1: Vector, scalar: float):
        return Vector(vector1.x + scalar, vector1.y + scalar, vector1.z + scalar)

    @staticmethod
    def multiply(vector1: Vector, scalar: float):
        return Vector(vector1.x * scalar, vector1.y * scalar, vector1.z * scalar)

    @staticmethod
    def divide(vector1: Vector, scalar: float):
        return Vector(vector1.x / scalar, vector1.y / scalar, vector1.z / scalar)

    @staticmethod
    def negate(vector1: Vector):
        return Vector(-vector1.x, -vector1.y, -vector1.z)

    @staticmethod
    def fromTo(vector1: Vector, vector2: Vector):
        return VectorFunctions.subtract(vector2, vector1)

    @staticmethod
    def dotProduct(vector1: Vector, vector2: Vector):           #SkalarProdukt
        return vector1.x*vector2.x + vector1.y*vector2.y + vector1.z*vector2.z

    @staticmethod
    def normalize(vector1: Vector):           #SkalarProdukt
        length = np.sqrt(np.power(vector1.x,2) + np.power(vector1.y,2) + np.power(vector1.z,2))
        return Vector(vector1.x/length,vector1.y/length, vector1.z/length)

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
        vectorFromCamera = VectorFunctions.fromTo(camera.center,self.point)
        vectorToSphereCenter = VectorFunctions.fromTo(self.point, sphere.center)

        dotProd1 = VectorFunctions.dotProduct(vectorFromCamera, vectorFromCamera)
        dotProd2 = VectorFunctions.dotProduct(vectorFromCamera, vectorToSphereCenter)
        dotProd3 = VectorFunctions.dotProduct(vectorToSphereCenter, vectorToSphereCenter)

        poly = np.poly1d([dotProd1, 2 * dotProd2, dotProd3 - sphere.r * sphere.r])
        result = poly.r[0]

        if (not np.iscomplex(result)): #Wenn er im Ersten Punkt schneidet reichts
            intersecting_point = Point(*VectorFunctions.add(VectorFunctions.multiply(vectorFromCamera,result),camera.center))
            normal_vector = VectorFunctions.fromTo(sphere.center, intersecting_point)
            normalized_normal_vector = VectorFunctions.normalize(normal_vector)
            color_vector = VectorFunctions.divide(normalized_normal_vector, 2)
            color_vector = VectorFunctions.addScalarToAll(color_vector, 0.5)
            pixel_color = Color(int(color_vector.x * 255), int(color_vector.y * 255), int(color_vector.z * 255))
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
