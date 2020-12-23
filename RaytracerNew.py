import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math

class Light():
    def __init__(self, coords):
        self.coords = coords

class Camera():
    def __init__(self, coords):
        self.coords = coords

class Screen():
    def __init__(self, coordsBotLeft, coordsBotRight, coordsTopRight, xResolution, yResolution):
        self.coordsBotLeft = coordsBotLeft
        self.coordsBotRight = coordsBotRight
        self.coordsTopRight = coordsTopRight
        self.xResolution = xResolution
        self.yResolution = yResolution

class Picture():
    def __init__(self, width, height):
        self.values = np.array(np.zeros((height,width,3), dtype=int))
        self.width = width
        self.height = height

    def insertPixel(self, color, x, y):
        self.values[y][x] = color

    def draw(self):
        plt.imshow(self.values)
        plt.gca().invert_yaxis()
        plt.show()

class Triangle():
    def __init__(self, p1, p2, p3, color):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.color = np.array(color)
        self.normalVector = np.cross(np.subtract(p2,p1),np.subtract(p3,p1))

    def intersects(self, source, ray):      #Möller Trumbore

        edge = np.subtract(self.p2, self.p1)
        edge2 = np.subtract(self.p3, self.p1)

        pvec = np.cross(ray,edge2)
        det = np.dot(edge, pvec)            #Check if Triangle is parallel to ray
        if det == 0:
            return False

        inv_det = 1/det
        tvec = np.subtract(source, self.p1)
        u = np.dot(tvec, pvec) * inv_det
        if u < 0 or u > 1:
            return False


        qvec = np.cross(tvec, edge)
        v = np.dot(ray,qvec) * inv_det
        if v < 0 or (v + u) > 1:
            return False

        t = np.dot(edge2, qvec) * inv_det
        if t >= 0:
            return np.add(source, ray*t)
        else:
            return False

    def lightIntensity(self, intersecPoint, lightSource):
        vec = np.subtract(lightSource, intersecPoint)
        factor = np.dot(vec,self.normalVector) / ((np.linalg.norm(vec) * np.linalg.norm(self.normalVector)))
        if factor < 0:
            factor = -factor
        return self.color * factor

class Rectangle:
    def __init__(self, coordsBotLeft, coordsBotRight, coordsTopRight, color):
        coordsTopLeft = coordsTopRight + np.subtract(coordsBotLeft, coordsBotRight)
        self.triangles = [Triangle(coordsBotLeft,coordsBotRight,coordsTopRight, color), Triangle(coordsBotLeft,coordsTopLeft,coordsTopRight, color)]

    def intersects(self, source, ray):
        for tri in self.triangles:
            intersecPoint = tri.intersects(source,ray)
            if np.any(intersecPoint!=0):
                return intersecPoint
        return False

    def lightIntensity(self, intersecPoint, lightSource):
        return self.triangles[0].lightIntensity(intersecPoint, lightSource)

class Cuboid:
    def __init__(self, coordsBotLeftFront, coordsBotRightFront, coordsTopRightFront, coordsTopRightBack, color):
        self.rectangles = []
        coordsTopLeftFront = coordsTopRightFront + np.subtract(coordsBotLeftFront, coordsBotRightFront)
        coordsBotLeftBack = coordsBotLeftFront + np.subtract(coordsTopRightBack, coordsTopRightFront)
        coordsBotRightBack = coordsBotRightFront + np.subtract(coordsTopRightBack, coordsTopRightFront)
        coordsTopLeftBack = coordsTopLeftFront + np.subtract(coordsTopRightBack, coordsTopRightFront)

        self.rectangles.append(Rectangle(coordsBotLeftFront, coordsBotRightFront, coordsTopRightFront, color))     #vorne und hinten
        self.rectangles.append(Rectangle(coordsBotLeftBack, coordsBotRightBack, coordsTopRightBack, color))

        self.rectangles.append(Rectangle(coordsBotLeftFront, coordsTopLeftFront, coordsTopLeftBack, color))        #links und rechts
        self.rectangles.append(Rectangle(coordsBotRightFront, coordsBotRightBack, coordsTopRightBack, color))

        self.rectangles.append(Rectangle(coordsBotLeftFront, coordsBotRightFront, coordsBotRightBack, color))      #oben und unten
        self.rectangles.append(Rectangle(coordsTopLeftFront, coordsTopRightFront, coordsTopRightBack, color))

        self.rectDict = {}

    def intersects(self, source, ray):
        realIntersecPoint = (math.inf,math.inf,math.inf)
        for rect in self.rectangles:
            intersecPoint = rect.intersects(source,ray)
            if np.any(intersecPoint!=0):
                self.rectDict[np.array2string(intersecPoint)] = rect
                if np.linalg.norm(np.subtract(intersecPoint,source)) < np.linalg.norm(np.subtract(realIntersecPoint, source)):
                    realIntersecPoint = intersecPoint

        if realIntersecPoint[0] == math.inf:
            return False
        return realIntersecPoint

    def lightIntensity(self, intersecPoint, lightSource):
        return self.rectDict[np.array2string(intersecPoint)].lightIntensity(intersecPoint,lightSource)


rect = Cuboid((0, 0, 4), (1,0,4), (4,8,4), (8,12,16), (255,0,0))

camera = Camera((2,3,0))
screen = Screen((0,1,1),(4,1,1),(4,5,1), 200, 200)
light = Light((10,10,0))

pic = Picture(screen.xResolution, screen.yResolution)



starttime = time.time()
for i in range(screen.xResolution):
    for j in range(screen.yResolution):
        widthVec = np.subtract(screen.coordsBotRight, screen.coordsBotLeft)
        screenWidth = np.linalg.norm(widthVec)
        heightVec = np.subtract(screen.coordsTopRight, screen.coordsBotRight)
        screenHeigth = np.linalg.norm(heightVec)

        pixelCoords = screen.coordsBotLeft + widthVec*i/screen.xResolution + heightVec*j/screen.yResolution
        ray = np.subtract(pixelCoords,camera.coords)
        intersecPoint = rect.intersects(camera.coords, ray)
        if np.any(intersecPoint!=0):
            color = rect.lightIntensity(intersecPoint, light.coords)
            pic.insertPixel(color, i, j)

        else:
            pic.insertPixel((175,238,238),i,j)

print(time.time()-starttime)

x =(0,0,0)
if x:
    print("hi")
else:
    print("bye")
pic.draw()

