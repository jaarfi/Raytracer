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

    def intersect(self, rayOrigin, rayDirections, writeToDict):
        rayDirections = np.array([ray/np.linalg.norm(ray) for ray in rayDirections])
        a = -(np.dot(rayOrigin, self.normalVector) + self.distanceToOrigin)
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b==0,-1,b)

        return a/b

    def colorsInPoints(self, intersectionPoints, lightSource):
        raysToLightSource = np.subtract(lightSource, intersectionPoints)
        dotProds = np.dot(raysToLightSource, self.normalVector)
        lengths1 = np.square(raysToLightSource[:,0]) + np.square(raysToLightSource[:,1]) + np.square(raysToLightSource[:,2])
        lengths1 = np.sqrt(lengths1)
        lengths = lengths1 * lengths1 * 0.1 * np.linalg.norm(self.normalVector)
        factors = np.divide(dotProds,lengths)

        color = (self.color,) * len(intersectionPoints)
        color = color * factors[:,None]
        return color

    def getNormalVector(self, intersectionPoint):
        return self.normalVector

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


    def intersect(self, rayOrigin, rayDirections, writeToDict):
        rayOrigin4 = np.append(rayOrigin,1)
        rayOriginBoxSpace = np.matmul(self.transformationMatrix, rayOrigin4)[:3]
        rayDirections4 = np.insert(np.array(rayDirections), 3, 0, axis=1)
        rayDirectionsBoxSpace = np.einsum('...ij,...j', self.transformationMatrix, rayDirections4)
        rayDirectionsBoxSpace = np.delete(rayDirectionsBoxSpace,3,1)
        rayDirectionsBoxSpace = np.array([ray/np.linalg.norm(ray) for ray in rayDirectionsBoxSpace])

        rayDirectionsBoxSpaceNoZeros = np.array([np.where(a==0,0.000001,a) for a in rayDirectionsBoxSpace])

        t1 = [np.divide((-rayOriginBoxSpace + self.size),a) for a in rayDirectionsBoxSpaceNoZeros]
        t2 = [np.divide((-rayOriginBoxSpace - self.size),a) for a in rayDirectionsBoxSpaceNoZeros]


        t3 = ([np.where(a<b,a,b) for a, b in np.array(list(zip(t1,t2)))])
        t4 = ([np.where(a>b,a,b) for a, b in np.array(list(zip(t1,t2)))])

        tMin = np.array([np.max(a) for a in t3])
        tMax = np.array([np.min(b) for b in t4])

        tReturn = np.where(np.logical_and(tMin<tMax, tMax>0),tMin,-1)

        tReturn = np.where(tMin<0,tMax,tReturn)

        if writeToDict:
            tReturnOnlyValidValues = tReturn[tReturn >= 0]
            rayDirectionsBoxSpaceOnlyHit = rayDirectionsBoxSpace[tReturn >= 0]

            t = np.array([(a[0],b[0],a[1],b[1],a[2],b[2]) for a,b in zip(t1,t2)])
            tOnlyValidValues = t[tReturn >= 0]

            tSides = [np.where(a == b)[0] for a,b in zip(tOnlyValidValues, tReturnOnlyValidValues)]
            tSides = [a[0] for a in tSides]



            normalVectors = np.array((  self.transformationMatrix[0][:3],           #rechts
                                        - 1* self.transformationMatrix[0][:3],      #links
                                        self.transformationMatrix[1][:3],           #oben
                                        -1 * self.transformationMatrix[1][:3],      #unten
                                        self.transformationMatrix[2][:3],           #//hinten
                                        -1 * self.transformationMatrix[2][:3]))     #vorne



            #intersecPointsBoxSpace = rayOriginBoxSpace + tReturnOnlyValidValues[...,None]*rayDirectionsBoxSpaceOnlyHit
            #intersecPointsBoxSpace4 = np.insert(np.array(intersecPointsBoxSpace), 3, 1, axis=1)
            #intersecPointsWorldSpace = np.einsum('...ij,...j', self.inverseTransformationMatrix, intersecPointsBoxSpace4)
            #intersecPointsWorldSpace = np.delete(intersecPointsWorldSpace,3,1)

            rayDirectionsOnlyHit = rayDirections[tReturn >= 0]
            intersecPointsWorldSpace = rayOrigin + tReturnOnlyValidValues[...,None]*rayDirectionsOnlyHit


            for i, val in enumerate(intersecPointsWorldSpace):
                self.normalVectorDict[np.array2string(np.around(val,5))] = normalVectors[tSides[i]]


        return tReturn

    def colorsInPoints(self, intersectionPoints, lightSource):
        normalVectors = []
        for i in intersectionPoints:
            normalVectors.append(self.normalVectorDict[np.array2string(np.around(i,5))])


        raysToLightSource = np.subtract(lightSource, intersectionPoints)
        dotProds = [np.dot(rayToLightSource,normalVector) for rayToLightSource, normalVector in zip(raysToLightSource,normalVectors)]
        lengths1 = np.square(raysToLightSource[:, 0]) + np.square(raysToLightSource[:, 1]) + np.square(
            raysToLightSource[:, 2])
        lengths1 = np.sqrt(lengths1)
        lengths = [length1 * length1 * 0.1 * np.linalg.norm(normalVector) for length1, normalVector in zip(lengths1,normalVectors)] # *lengths*0.1 for fsitance to light
        factors = np.divide(dotProds, lengths)
        color = (self.color,) * len(intersectionPoints)
        color = color * factors[:, None]
        color = np.abs(color)
        color = color.astype(int)

        return color

    def getNormalVector(self, intersectionPoint):
        return self.normalVectorDict[np.array2string(np.around(intersectionPoint,5))]

class Background:
    def __init__(self, color):
        self.color = color

def calculateTrueDistances(trueDistances, distances):
    trueDistancesValues, trueDistancesObjects = zip(*trueDistances)
    distancesValues, distancesObjects = zip(*distances)

    trueDistancesValues = np.array(trueDistancesValues)
    distancesValues = np.array(distancesValues)

    zeros = np.zeros(len(distancesValues))

    trueDistances = np.where(np.logical_and((zeros <= distancesValues),(distancesValues < trueDistancesValues))[..., None], distances, trueDistances)

    return trueDistances




def rayTrace(allPixelCoords, cameraCoords, scene, lightSource):
    pixelRays = np.subtract(allPixelCoords, cameraCoords)
    pixelRays = np.array([pixelRay/np.linalg.norm(pixelRay) for pixelRay in pixelRays])
    trueDistances = np.array(((maxDistance,background),)*len(allPixelCoords))

    for body in scene:
        distances = body.intersect(cameraCoords, pixelRays, True)
        distancesAndObject = np.array(list(zip(distances, np.repeat(body,len(distances)))))
        trueDistances = calculateTrueDistances(trueDistances,distancesAndObject)

    trueIntersectionsPointsValues, trueIntersectionsPointsObjects = zip(*trueDistances)
    trueIntersectionsPointsValues = np.array(trueIntersectionsPointsValues)
    trueIntersectionsPointsValues = cameraCoords + pixelRays * trueIntersectionsPointsValues[:, None]
    trueIntersectionPoints = np.array(list(zip(trueIntersectionsPointsValues, trueIntersectionsPointsObjects)))

    displacedIntersectionPoints = np.array([point + 0.001 * body.getNormalVector(np.around(point,5)) for point, body in trueIntersectionPoints])
    raysFromLightToPoint = np.subtract(displacedIntersectionPoints, lightSource)
    distancesToLight = np.array([np.linalg.norm(ray) for ray in raysFromLightToPoint])
    trueShadowDistances = np.array(list(zip(distancesToLight, np.ones(len(distancesToLight)))))

    for body in scene:
        distances = body.intersect(lightSource, raysFromLightToPoint, False)
        distancesAndFactor = np.array(list(zip(distances, np.repeat(0.25,len(distances)))))
        trueShadowDistances = calculateTrueDistances(trueShadowDistances,distancesAndFactor)

    dummyArray, factorsArray = zip(*trueShadowDistances)




    colors = np.arange(xResolution*yResolution)
    colors = np.ones(3) * colors[:,None]

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
    colors = np.array([color * factorsArray[i] for i, color in enumerate(colors)])

    colors = colors.astype(int)
    return colors



hinten  = Plane((0, 0, -1), 10, (120,120,120))
rechts  = Plane((-1, 0, 0), 10, (0,0,120))
unten   = Plane((0,1,0), 0, (120,120,120))
oben    = Plane((0,-1,0),10,(120,120,120))
links   = Plane((1,0,0),0,(120,0,0))
cube1   = Cuboid((1.5,1.5,1.5),(2.5,1.5,4),0,(0,120,120))
cube2   = Cuboid((1.5,3,1.5),(7,3,5),45,(120,120,120))


scene = [rechts, hinten, links, unten, oben, cube1, cube2]

xResolution = 1080
yResolution = 1080
maxDistance = 1e6

camera = (5, 5, -1)
screen = ((0, 0), (10, 10))
light = (5, 9, 5)
background = Background((0,0,0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0],xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1],yResolution)

pixelCoords = [(x,y,0) for y in pixelCoordsY for x in pixelCoordsX ]

print("los gehts")
starttime = time.time()

colors = rayTrace(pixelCoords, camera, scene, light)


print("It took: ", time.time() - starttime, "s") #50x50 = 9.2s 100x100 = 36.9


colors = np.reshape(colors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()


