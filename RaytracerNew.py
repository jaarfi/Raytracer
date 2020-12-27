import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math


class Light():
    def __init__(self, coords, strength):
        self.coords = coords
        self.strength = strength


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
        self.values = np.array(np.zeros((height, width, 3), dtype=int))
        self.width = width
        self.height = height

    def insertPixel(self, color, x, y):
        self.values[y][x] = color

    def draw(self):
        plt.imshow(self.values)
        plt.gca().invert_yaxis()
        plt.show()


class Triangle():
    def __init__(self, p1, p2, p3, color, name):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.color = np.array(color)
        self.normalVector = np.cross(np.subtract(p3, p1), np.subtract(p2, p1))
        self.intPoint = ()
        self.name = name

    def intersects(self, source, ray):  # Möller Trumbore

        edge = np.subtract(self.p2, self.p1)
        edge2 = np.subtract(self.p3, self.p1)

        pvec = np.cross(ray, edge2)
        det = np.dot(edge, pvec)  # Check if Triangle is parallel to ray
        if det == 0:
            return False

        inv_det = 1 / det
        tvec = np.subtract(source, self.p1)
        u = np.dot(tvec, pvec) * inv_det
        if u < 0 or u > 1:
            return False

        qvec = np.cross(tvec, edge)
        v = np.dot(ray, qvec) * inv_det
        if v < 0 or (v + u) > 1:
            return False

        t = np.dot(edge2, qvec) * inv_det
        if t >= 0:
            return np.add(source, ray * t)
        else:
            return False

    def colorInPoint(self, intersecPoint, lightSource):
        vec = np.subtract(lightSource, intersecPoint)
        length = np.linalg.norm(vec)
        factor = np.dot(vec, self.normalVector) / (length * 0.1 * length * np.linalg.norm(self.normalVector))
        if factor < 0:
            factor = -factor
        return self.color * factor

    def getNormalVector(self, intersecPoint):
        return self.normalVector


class Rectangle:
    def __init__(self, coordsBotLeft, coordsBotRight, coordsTopRight, color, name):
        coordsTopLeft = coordsTopRight + np.subtract(coordsBotLeft, coordsBotRight)
        self.triangles = [Triangle(coordsBotLeft, coordsBotRight, coordsTopRight, color, name),
                          Triangle(coordsBotLeft, coordsTopLeft, coordsTopRight, color, name)]
        self.normalVector = self.triangles[0].normalVector
        self.intPoint = ()
        self.name = name

    def intersects(self, source, ray):
        for tri in self.triangles:
            intersecPoint = tri.intersects(source, ray)
            if np.any(intersecPoint != 0):
                return intersecPoint
        return False

    def colorInPoint(self, intersecPoint, lightSource):
        return self.triangles[0].colorInPoint(intersecPoint, lightSource)

    def getNormalVector(self, intersecPoint):
        return self.normalVector


class Cuboid:
    def __init__(self, coordsBotLeftFront, coordsBotRightFront, coordsTopRightFront, coordsTopRightBack, color, name):
        self.rectangles = []
        coordsTopLeftFront = coordsTopRightFront + np.subtract(coordsBotLeftFront, coordsBotRightFront)
        coordsBotLeftBack = coordsBotLeftFront + np.subtract(coordsTopRightBack, coordsTopRightFront)
        coordsBotRightBack = coordsBotRightFront + np.subtract(coordsTopRightBack, coordsTopRightFront)
        coordsTopLeftBack = coordsTopLeftFront + np.subtract(coordsTopRightBack, coordsTopRightFront)

        self.rectangles.append(
            Rectangle(coordsBotLeftFront, coordsTopLeftFront, coordsTopRightFront, color, name))  # vorne
        self.rectangles.append(
            Rectangle(coordsBotRightBack, coordsTopRightBack, coordsTopLeftBack, color, name))  # hinten

        self.rectangles.append(
            Rectangle(coordsBotLeftBack, coordsBotLeftFront, coordsTopLeftFront, color, name))  # links
        self.rectangles.append(
            Rectangle(coordsBotRightFront, coordsBotRightBack, coordsTopRightBack, color, name))  # rechts

        self.rectangles.append(
            Rectangle(coordsTopLeftFront, coordsTopRightFront, coordsTopRightBack, color, name))  # oben
        self.rectangles.append(
            Rectangle(coordsBotLeftBack, coordsBotRightBack, coordsBotRightFront, color, name))  # unten

        self.rectDict = {}

        self.intPoint = ()
        self.name = name

    def intersects(self, source, ray):
        realIntersecPoint = (math.inf, math.inf, math.inf)
        for rect in self.rectangles:
            intersecPoint = rect.intersects(source, ray)
            if np.any(intersecPoint != 0):
                self.rectDict[np.array2string(intersecPoint)] = rect
                if np.linalg.norm(np.subtract(intersecPoint, source)) < np.linalg.norm(
                        np.subtract(realIntersecPoint, source)):
                    realIntersecPoint = intersecPoint

        if realIntersecPoint[0] == math.inf:
            return False
        return realIntersecPoint

    def colorInPoint(self, intersecPoint, lightSource):
        return self.rectDict[np.array2string(intersecPoint)].colorInPoint(intersecPoint, lightSource)

    def getNormalVector(self, intersecPoint):
        return self.rectDict[np.array2string(intersecPoint)].normalVector


def traceRay(pixelCoords, cameraCoords, scene, light):
    ray = np.subtract(pixelCoords, camera.coords)
    dicto = {}
    tempdist = np.inf
    for s in scene:
        intersecPoint = s.intersects(camera.coords, ray)
        vec = np.subtract(intersecPoint, cameraCoords)
        if np.any(intersecPoint != 0):
            dist = np.linalg.norm(vec)
            if dist < tempdist:
                dicto[dist] = s
                tempdist = dist
                s.intPoint = intersecPoint

    if tempdist == np.inf:
        return (0, 0, 0)

    hitObject = dicto[tempdist]
    rayFromHitPointtoLight = np.subtract(light.coords, hitObject.intPoint)
    outerHitPoint = hitObject.intPoint + hitObject.getNormalVector(hitObject.intPoint) * 0.0001

    color = hitObject.colorInPoint(hitObject.intPoint, light.coords)

    for s in scene:
        shadowPoint = s.intersects(outerHitPoint, rayFromHitPointtoLight)
        if np.any(shadowPoint) != 0:
            if np.linalg.norm(np.subtract(shadowPoint, hitObject.intPoint)) < np.linalg.norm(rayFromHitPointtoLight):
                #print(hitObject.name, "trifft", s.name, "auf dem weg zum licht")
                return color * 0.75
    return color


scene = [Rectangle((0, 0, 0), (0, 0, 10), (0, 10, 10), (162, 0, 0), "links"),
         Rectangle((10, 0, 10), (10, 0, 0), (10, 10, 0), (0, 0, 162), "rechts"),
         Rectangle((0, 0, 0), (10, 0, 0), (10, 0, 10), (120, 120, 120), "unten"),
         Rectangle((0, 10, 10), (10, 10, 10), (10, 10, 0), (120, 120, 120), "oben"),
         Rectangle((0, 0, 10), (10, 0, 10), (10, 10, 10), (120, 120, 120), "hinten"),
         #Rectangle((0, 4, 5), (10, 4, 5), (10, 4, 10), (120, 120, 120), "mitte"),

         Cuboid((1, 0, 3), (5,0,3), (5,4,3), (5,4,7),(162,162,0),"1"),
         Cuboid((7, 0, 4), (9,0,6), (9,5,6), (7,5,8),(60,60,60),"2"),
         ]

camera = Camera((5, 5, -5))
screen = Screen((0, 0, 0), (0, 10, 0), (10, 10, 0), 200, 200)
light = Light((5, 9, 5), 1)

pic = Picture(screen.xResolution, screen.yResolution)

starttime = time.time()
for i in range(screen.xResolution):
    if i % 10 == 0:
        print(i / float(screen.xResolution) * 100, "%")
    for j in range(screen.yResolution):
        widthVec = np.subtract(screen.coordsBotRight, screen.coordsBotLeft)
        screenWidth = np.linalg.norm(widthVec)
        heightVec = np.subtract(screen.coordsTopRight, screen.coordsBotRight)
        screenHeigth = np.linalg.norm(heightVec)

        pixelCoords = screen.coordsBotLeft + widthVec * i / screen.xResolution + heightVec * j / screen.yResolution
        pic.insertPixel(traceRay(pixelCoords, camera.coords, scene, light), j, i)

print(time.time() - starttime)

x = (0, 0, 0)
if x:
    print("hi")
else:
    print("bye")
pic.draw()
