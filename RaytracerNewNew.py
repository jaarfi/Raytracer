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
        lengths1 = np.square(raysToLightSource[:,0]) + np.square(raysToLightSource[:,1]) + np.square(raysToLightSource[:,2])
        lengths1 = np.sqrt(lengths1)
        lengths = lengths1 * np.linalg.norm(self.normalVector) * lengths1 * 0.1             #*lengths*0.1 for fsitance to light
        factors = np.divide(dotProds,lengths)

        color = (self.color,) * len(intersectionPoints)
        color = color * factors[:,None]
        return color

class Cuboid:
    def __init__(self, size, center, rotation, color):
        self.size = size
        self.rotation = np.radians(rotation)
        self.center = center
        self.color = color
        self.rotationYMatrix = np.array([[np.cos(self.rotation), 0, np.sin(self.rotation), 0],
                                        [0,1,0,0],
                                        [-np.sin(self.rotation),0,np.cos(self.rotation),0],
                                        [0,0,0,1]])
        self.translationMatrix = np.array([[1,0,0,-center[0]],
                                           [0,1,0,-center[1]],
                                           [0,0,1,-center[2]],
                                           [0,0,0,1]])
        self.transformationMatrix = np.matmul(self.rotationYMatrix, self.translationMatrix)
        self.transformationMatrix = np.around(self.transformationMatrix, 3)
        self.inverseTransformationMatrix = np.linalg.inv(self.transformationMatrix)
        self.normalVectorDict = {}


    def intersect(self, rayOrigin, rayDirections):
        rayOrigin4 = np.append(rayOrigin,1)
        rayOriginBoxSpace = np.matmul(self.transformationMatrix, rayOrigin4)[:3]
        rayDirections4 = np.insert(np.array(rayDirections), 3, 0, axis=1)
        rayDirectionsBoxSpace = np.einsum('...ij,...j', self.transformationMatrix, rayDirections4)
        rayDirectionsBoxSpace = np.delete(rayDirectionsBoxSpace,3,1)

        rayDirectionsBoxSpaceNoZeros = np.array([np.where(a==0,0.000001,a) for a in rayDirectionsBoxSpace])

        t1 = [np.divide((-rayOriginBoxSpace + self.size),a) for a in rayDirectionsBoxSpaceNoZeros]
        t2 = [np.divide((-rayOriginBoxSpace - self.size),a) for a in rayDirectionsBoxSpaceNoZeros]


        t3 = ([np.where(a<b,a,b) for a, b in np.array(list(zip(t1,t2)))])
        t4 = ([np.where(a>b,a,b) for a, b in np.array(list(zip(t1,t2)))])

        tMin = np.array([np.max(a) for a in t3])
        tMax = np.array([np.min(b) for b in t4])

        tReturn = np.where(np.logical_and(tMin<tMax, tMax>0),tMin,-1)

        tReturnOnlyValidValues = tReturn[tReturn >= 0]
        #rayDirectionsBoxSpaceOnlyHit = np.where((tReturn > 0)[...,None], rayDirectionsBoxSpace, np.zeros((len(rayDirectionsBoxSpace),3)))
        rayDirectionsBoxSpaceOnlyHit = rayDirectionsBoxSpace[tReturn >= 0]

        t = np.array([(a[0],b[0],a[1],b[1],a[2],b[2]) for a,b in zip(t3,t4)])
        tOnlyValidValues = t[tReturn >= 0]

        #index = np.arange(6)
        tSides = [np.where(a == b)[0] for a,b in zip(tOnlyValidValues, tReturnOnlyValidValues)]
        tSides = [a[0] for a in tSides]



        normalVectors = np.array((  -1 * self.transformationMatrix[0][:3],
                                    self.transformationMatrix[0][:3],
                                    -1 * self.transformationMatrix[1][:3],
                                    self.transformationMatrix[1][:3],
                                    -1 * self.transformationMatrix[2][:3],
                                    self.transformationMatrix[2][:3]))

        print(normalVectors)
        intersecPointsBoxSpace = rayOriginBoxSpace + tReturnOnlyValidValues[...,None]*rayDirectionsBoxSpaceOnlyHit

        return tReturn

    def colorsInPoints(self, intersectionPoints, lightSource):
        color = (self.color,) * len(intersectionPoints)
        return color


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
    trueIntersectionsPointsValues = cameraCoords + pixelRays * trueIntersectionsPointsValues[:, None]
    trueIntersectionPoints = np.array(list(zip(trueIntersectionsPointsValues, trueIntersectionsPointsObjects)))

    colors = np.arange(xResolution*yResolution)
    colors = np.ones(3) * colors[:,None]
    #for x in trueIntersectionPoints:
    #    colors.append(x[1].color)

    allIndices = []
    allBodyColors = []
    for body in scene:
        indices = np.where(trueIntersectionPoints[:,1]==body)[0]
        bodyIntersectionsPoints = trueIntersectionPoints[trueIntersectionPoints[:,1]==body]

        bodyIntersectionsPointsValues, bodyIntersectionsPointsObjects = zip(*bodyIntersectionsPoints)
        bodyColors = body.colorsInPoints(bodyIntersectionsPointsValues, lightSource)
        allIndices.extend(indices)
        allBodyColors.extend(bodyColors)


    allIndices = np.array(allIndices)
    allBodyColors = np.array(allBodyColors)
    sorter = np.argsort(allIndices)


    colors = allBodyColors[sorter]

    colors = colors.astype(int)
    return colors



hinten  = Plane((0, 0, -1), 10, (120,120,120))
rechts  = Plane((-1, 0, 0), 10, (0,0,120))
unten   = Plane((0,1,0), 0, (120,120,120))
oben    = Plane((0,-1,0),10,(120,120,120))
links   = Plane((1,0,0),0,(120,0,0))
cube1   = Cuboid((1.5,1.5,1.5),(2,1.5,4),0,(120,120,0))
cube2   = Cuboid((1.5,3,1.5),(7,3,5),45,(0,102,204))


scene = [#Plane((0, 0, 0), (0, 0, 10), (0, 10, 10), (162, 0, 0), "links"),
         #Plane((10, 0, 10), (10, 0, 0), (10, 10, 0), (0, 0, 162), "rechts"),
         #Plane((0, 0, 0), (10, 0, 0), (10, 0, 10), (120, 120, 120), "unten")
         #Plane((0, 10, 10), (10, 10, 10), (10, 10, 0), (120, 120, 120), "oben"),
         #Plane((0, 0, 10), (10, 0, 10), (10, 10, 10), (120, 120, 120), "hinten")

         #AxisAlignedCuboid((1, 0, 3), (5,0,3), (5,4,3), (5,4,7),(162,162,0),"1")
         #Cuboid((7, 0, 4), (9,0,6), (9,5,6), (7,5,8),(60,60,60),"2"),
         ]

xResolution = 256
yResolution = 256
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
light = (5, 9, 5)
background = Background((0,0,0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0],xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1],yResolution)

pixelCoords = [(x,y,0) for y in pixelCoordsY for x in pixelCoordsX ]

print("los gehts")
starttime = time.time()

colors = rayTrace(pixelCoords, camera, [rechts, hinten, links, unten, oben, cube1, cube2], light)


print("It took: ", time.time() - starttime, "s") #50x50 = 9.2s 100x100 = 36.9


colors = np.reshape(colors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()


