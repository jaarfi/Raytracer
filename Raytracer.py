import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math




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
    def length(vector1: Vector):
        return np.sqrt(np.power(vector1.x,2) + np.power(vector1.y,2) + np.power(vector1.z,2))

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
    def angle(vector1: Vector, vector2: Vector):
        return VectorFunctions.dotProduct(vector1, vector2) / (VectorFunctions.length(vector1) * VectorFunctions.length(vector2))

    @staticmethod
    def normalize(vector1: Vector):
        length = VectorFunctions.length(vector1)
        return Vector(vector1.x/length,vector1.y/length, vector1.z/length)

class Color(typing.NamedTuple):
    r: int = 0
    g: int = 0
    b: int = 0

class HittableObject(typing.NamedTuple):
    def hitInPoints(self):
        pass

class Sphere(HittableObject):

    def __new__(cls, r: float, center: Point, color: Color):
        self = super(Sphere, cls).__new__(cls)
        self.r = r
        self.center = center
        self.color = color
        return self

    def hitInPoints(self, point: Point, vector: Vector):
        vectorToSphereCenter = VectorFunctions.fromTo(point, self.center)

        dotProd1 = VectorFunctions.dotProduct(vector, vector)
        dotProd2 = VectorFunctions.dotProduct(vector, vectorToSphereCenter)
        dotProd3 = VectorFunctions.dotProduct(vectorToSphereCenter, vectorToSphereCenter)

        poly = np.poly1d([dotProd1, 2 * dotProd2, dotProd3 - sphere.r * sphere.r])
        results = poly.r

        return results

class ViewingScreen(typing.NamedTuple):
    bottom_left: tuple
    width: int
    height: int
    resolution_x: int
    resolution_y: int

class Camera(typing.NamedTuple):
    center: Point
    viewingScreen: ViewingScreen

class LightSource(typing.NamedTuple):
    center: Point
    strength: float

class Pixel(typing.NamedTuple):
    point: Point

    def color(self, camera: Camera, sphere: Sphere, light: LightSource):
        vectorFromCamera = VectorFunctions.fromTo(camera.center, self.point)
        results = sphere.hitInPoints(self.point, vectorFromCamera)

        if (not np.iscomplex(results[0])): #Wenn er im Ersten Punkt schneidet reichts
            intersectingPoint = Point(*VectorFunctions.add(VectorFunctions.multiply(vectorFromCamera,results[0]),camera.center))
            normalVector = VectorFunctions.fromTo(sphere.center, intersectingPoint)
            normalizedNormalVector = VectorFunctions.normalize(normalVector)

            vectorToLight = VectorFunctions.fromTo(self.point, light.center)
            lightIntensity = math.pow(light.strength,VectorFunctions.length(vectorFromCamera))
            pixelColorIntensity = VectorFunctions.angle(vectorToLight, normalizedNormalVector) * lightIntensity

            colorAsVector = VectorFunctions.multiply(Vector(*sphere.color), abs(pixelColorIntensity))
            pixelColor = Color(int(colorAsVector.x), int(colorAsVector.y), int(colorAsVector.z))
        else:
            pixelColor = Color(240,255,255)  # Normale Farbe

        return pixelColor

class Main():
    def __init__(self, sphere: Sphere, camera: Camera, light: LightSource):
        self.screen = camera.viewingScreen
        self.windowWidth = self.screen.resolution_x
        self.windowHeight = self.screen.resolution_y
        self.picture = Picture(self.windowWidth, self.windowHeight)
        self.light = light

        self.calculatePicture()
        self.drawPicture()

    def calculatePicture(self):
        now = time.time()
        for i in range(1, self.windowWidth):
            for j in range(1, self.windowHeight):
                currentPixelPoint = Point(*VectorFunctions.add(self.screen.bottom_left, Vector(self.screen.width / self.windowWidth * i, self.screen.height / self.windowHeight * j, 0)))
                currentPixel = Pixel(currentPixelPoint)


                self.picture.insertPixel(currentPixel.color(camera,sphere,light), i, j)

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

camera = Camera(Point(0,0,0),ViewingScreen(Point(-2,-1,-4),4,4,512,512))
hit = HittableObject()
point = Point(0,1,-2)
color = Color(0,128,255)
sphere = Sphere(0.5, point, color)
light = LightSource(Point(-2, 2, 1),0.99)
main = Main(sphere, camera, light)
v = Vector(0,0,0)
