import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math


class Plane:
    def __init__(self, normalVector, distanceToOrigin, color):
        self.normalVector = normalVector / np.linalg.norm(normalVector)
        self.distanceToOrigin = distanceToOrigin
        self.color = color

    def intersect(self, rayOrigin, rayDirections):
        rayDirections = rayDirections/np.linalg.norm(rayDirections)
        a = -(np.dot(rayOrigin, self.normalVector) + self.distanceToOrigin)
        b = np.dot(rayDirections, self.normalVector)
        return a/b

    def colorsInPoints(self, intersectionPoints, lightSource):
        raysToLightSource = np.subtract(lightSource, intersectionPoints)
        dotProds = np.dot(raysToLightSource, self.normalVector)
        lengths = np.linalg.norm(raysToLightSource) * np.linalg.norm(self.normalVector)
        print("len rays:", len(raysToLightSource), "len Dotprods:", len(dotProds), "lengths", lengths)
        factors = dotProds / lengths
        color = (self.color,)*len(intersectionPoints)
        #color = color * factors[:,None]
        return color

class AxisAlignedCuboid:
    def __init__(self):
        pass

class Background:
    def __init__(self, color):
        self.color = color

def calculateTrueDistances(trueDistances, distances, body):
    trueDistancesValues, trueDistancesObjects = zip(*trueDistances)
    distancesValues, distancesObjects = zip(*distances)

    trueDistancesValues = np.array(trueDistancesValues)
    distancesValues = np.array(distancesValues)



    #trueDistancesObjects = np.where(distancesValues < trueDistancesValues, distancesObjects, trueDistancesObjects)
    #trueDistancesValues = np.where(distancesValues < trueDistancesValues, distancesValues, trueDistancesValues)



    #trueDistances = np.array(list(zip(trueDistancesValues, trueDistancesObjects)))

    zeros = np.zeros(len(distancesValues))

    trueDistances = np.where(np.logical_and((zeros <= distancesValues),(distancesValues < trueDistancesValues))[..., None], distances, trueDistances)

    return trueDistances


def rayTrace(allPixelCoords, cameraCoords, scene, lightSource):
    pixelRays = np.subtract(allPixelCoords, cameraCoords)
    pixelRays = pixelRays/np.linalg.norm(pixelRays)
    trueDistances = np.array(((maxDistance,background),)*len(allPixelCoords))
    for body in scene:
        distances = body.intersect(cameraCoords, pixelRays)
        distancesAndObject = np.array(list(zip(distances, np.repeat(body,len(distances)))))
        trueDistances = calculateTrueDistances(trueDistances,distancesAndObject,body)

    trueIntersectionsPointsValues, trueIntersectionsPointsObjects = zip(*trueDistances)
    trueIntersectionsPointsValues = np.array(trueIntersectionsPointsValues)
    trueIntersectionsPointsValues = pixelRays * trueIntersectionsPointsValues[:, None]
    trueIntersectionPoints = np.array(list(zip(trueIntersectionsPointsValues, trueIntersectionsPointsObjects)))

    colors = np.arange(xResolution*yResolution)
    colors = np.ones(3) * colors[:,None]
    #for x in trueIntersectionPoints:
    #    colors.append(x[1].color)

    allIndices = []
    allBodyColors = []
    for body in scene:
        indices = np.where(trueIntersectionPoints[:,1]==body)[0]
        #indices = np.ones(3) * indices[:,None]
        bodyIntersectionsPoints = trueIntersectionPoints[trueIntersectionPoints[:,1]==body]

        bodyIntersectionsPointsValues, bodyIntersectionsPointsObjects = zip(*bodyIntersectionsPoints)
        #print(bodyIntersectionsPointsValues)
        #bodyIntersectionsPoints = np.array(list(zip(bodyIntersectionsPointsValues, indices)))
        #print(bodyIntersectionsPoints)
        bodyColors = body.colorsInPoints(bodyIntersectionsPointsValues, lightSource)
        #for i, val in enumerate(indices):
        #    print( colors[val], bodyColors[i])
        #    colors[val] = bodyColors[i]
        allIndices.extend(indices)
        allBodyColors.extend(bodyColors)

    #colors = np.where(colors == allIndices, allBodyColors, colors)
    for i, val in enumerate(allIndices):
        colors[val] = allBodyColors[i]
    #print(allIndices)
    #colors = allBodyColors[allIndices]
    #print(colors)
    colors = colors.astype(int)
    return colors



hinten  = Plane((0, 0, -1), 10, (120,120,120))
rechts  = Plane((-1, 0, 0), 10, (0,0,255))
unten   = Plane((0,1,0), 0, (120,120,120))
oben    = Plane((0,-1,0),10,(120,120,120))
links   = Plane((1,0,0),0,(255,0,0))


scene = [#Plane((0, 0, 0), (0, 0, 10), (0, 10, 10), (162, 0, 0), "links"),
         #Plane((10, 0, 10), (10, 0, 0), (10, 10, 0), (0, 0, 162), "rechts"),
         #Plane((0, 0, 0), (10, 0, 0), (10, 0, 10), (120, 120, 120), "unten")
         #Plane((0, 10, 10), (10, 10, 10), (10, 10, 0), (120, 120, 120), "oben"),
         #Plane((0, 0, 10), (10, 0, 10), (10, 10, 10), (120, 120, 120), "hinten")

         #AxisAlignedCuboid((1, 0, 3), (5,0,3), (5,4,3), (5,4,7),(162,162,0),"1")
         #Cuboid((7, 0, 4), (9,0,6), (9,5,6), (7,5,8),(60,60,60),"2"),
         ]

xResolution = 150
yResolution = 150
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
light = (5, 9, 5)
background = Background((0,0,0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0],xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1],yResolution)

pixelCoords = [(x,y,0) for y in pixelCoordsY for x in pixelCoordsX ]


starttime = time.time()

colors = rayTrace(pixelCoords, camera, [rechts, hinten, links, unten, oben], light)


print("It took: ", time.time() - starttime, "s") #50x50 = 9.2s 100x100 = 36.9

colors = np.reshape(colors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()

