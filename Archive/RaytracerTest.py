import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math


class Light:
    def __init__(self, origin, ambient, diffuse, specular, halfLength):
        self.origin = np.array(origin)
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.halfLength = halfLength

        tempArray = np.array([[halfLength,0,halfLength],[-halfLength,0,halfLength],[halfLength,0,-halfLength],[-halfLength,0,-halfLength]])
        self.vertices = [origin+tempArray][0]


class Plane:
    def __init__(self, normalVector, distanceToOrigin, colorAmbient, colorDiffuse, colorSpecular, shininess,
                 reflection):
        self.normalVector = normalVector / np.linalg.norm(normalVector)
        self.distanceToOrigin = distanceToOrigin
        self.colorAmbient = colorAmbient
        self.colorDiffuse = colorDiffuse
        self.colorSpecular = colorSpecular
        self.shininess = shininess
        self.reflection = reflection

    def intersect(self, rayOrigins, rayDirections, writeToDict):
        # print(rayOrigins)
        rayDirections = np.array([ray / np.linalg.norm(ray) for ray in rayDirections])
        # a = [np.dot(ray,self.normalVector) for ray in rayOrigins]
        # print(a)
        # a = -(a+self.distanceToOrigin)
        a = np.array([-(np.dot(ray, self.normalVector) + self.distanceToOrigin) for ray in rayOrigins])
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b == 0, -1, b)
        # print("a: ", a, "b: ",b)
        # print("a/b: ",a/b)

        return a / b

    def colorsInPoints(self, intersectionPoints, light, camera, currentDepth):
        lightSource = light.origin
        raysToLightSource = np.subtract(lightSource, intersectionPoints)
        raysToLightSource = np.array([ray / np.linalg.norm(ray) for ray in raysToLightSource])
        dotProdsDiffuse = np.array([np.dot(ray, self.normalVector) for ray in raysToLightSource])
        raysToCamera = np.subtract(camera, intersectionPoints)
        raysToCamera = np.array([ray / np.linalg.norm(ray) for ray in raysToCamera])
        addedRays = raysToLightSource + raysToCamera
        addedRays = np.array([ray / np.linalg.norm(ray) for ray in addedRays])
        dotProdsSpecular = np.array([np.dot(ray, self.normalVector) for ray in addedRays])
        dotProdsSpecular = np.power(dotProdsSpecular, self.shininess / 4)

        ambient = self.colorAmbient * light.ambient
        ambient = (ambient,) * len(intersectionPoints)
        diffuse = self.colorDiffuse * light.diffuse
        diffuse = (diffuse,) * len(intersectionPoints)
        diffuse = np.array([a * b for a, b in zip(diffuse, dotProdsDiffuse)])
        specular = self.colorSpecular * light.specular
        specular = (specular,) * len(intersectionPoints)
        specular = np.array([a * b for a, b in zip(specular, dotProdsSpecular)])

        colors = ambient + diffuse + specular
        return colors

    def colorInPoint(self, intersectionPoint, light, camera):
        lightSource = light.origin

        rayToLightSource = np.subtract(lightSource, intersectionPoint)
        rayToLightSource = rayToLightSource / np.linalg.norm(rayToLightSource)
        dotProdDiffuse = np.dot(rayToLightSource, self.normalVector)

        rayToCamera = np.subtract(camera, intersectionPoint)
        rayToCamera = rayToCamera / np.linalg.norm(rayToCamera)

        addedRay = rayToLightSource + rayToCamera
        addedRay = addedRay / np.linalg.norm(addedRay)

        dotProdSpecular = np.dot(addedRay, self.normalVector)
        dotProdSpecular = np.power(dotProdSpecular, self.shininess / 4)

        ambient = self.colorAmbient * light.ambient
        diffuse = self.colorDiffuse * light.diffuse * dotProdDiffuse
        specular = self.colorSpecular * light.specular * dotProdSpecular

        color = ambient + diffuse + specular
        return color

    def getNormalVector(self, intersectionPoint):
        return self.normalVector

    def getReflectedRay(self, intersectionPoint, rayDirection):
        # print("isec: ",intersectionPoint, "dir: ", rayDirection)
        rayDirection = rayDirection / np.linalg.norm(rayDirection)
        normalVec = self.normalVector
        temp = 2 * np.multiply(np.dot(normalVec, rayDirection), normalVec)
        return rayDirection - temp


class Cuboid:
    def __init__(self, size, center, rotation, colorAmbient, colorDiffuse, colorSpecular, shininess, reflection):
        self.size = size
        self.rotation = np.radians(rotation)
        self.center = center

        self.colorAmbient = colorAmbient
        self.colorDiffuse = colorDiffuse
        self.colorSpecular = colorSpecular
        self.shininess = shininess
        self.rotationYMatrix = np.array([[np.cos(self.rotation), 0, np.sin(self.rotation), 0],
                                         [0, 1, 0, 0],
                                         [-np.sin(self.rotation), 0, np.cos(self.rotation), 0],
                                         [0, 0, 0, 1]])
        self.translationMatrix = np.array([[1, 0, 0, -center[0]],
                                           [0, 1, 0, -center[1]],
                                           [0, 0, 1, -center[2]],
                                           [0, 0, 0, 1]])
        self.transformationMatrix = np.matmul(self.rotationYMatrix, self.translationMatrix)
        self.transformationMatrix = np.around(self.transformationMatrix, 3)
        self.inverseTransformationMatrix = np.linalg.inv(self.transformationMatrix)
        self.normalVectorDict = {}
        self.reflection = reflection

    def intersect(self, rayOrigins, rayDirections, writeToDict):
        # rayOrigin4 = np.append(rayOrigin,1)
        # rayOriginBoxSpace = np.matmul(self.transformationMatrix, rayOrigin4)[:3]
        rayDirections4 = np.insert(np.array(rayDirections), 3, 0, axis=1)
        rayDirectionsBoxSpace = np.einsum('...ij,...j', self.transformationMatrix, rayDirections4)
        rayDirectionsBoxSpace = np.delete(rayDirectionsBoxSpace, 3, 1)
        rayDirectionsBoxSpace = np.array([ray / np.linalg.norm(ray) for ray in rayDirectionsBoxSpace])

        rayOrigins4 = np.insert(np.array(rayOrigins), 3, 1, axis=1)
        rayOriginsBoxSpace = np.einsum('...ij,...j', self.transformationMatrix, rayOrigins4)
        rayOriginsBoxSpace = np.delete(rayOriginsBoxSpace, 3, 1)

        # print(rayOriginsBoxSpace)

        rayDirectionsBoxSpaceNoZeros = np.array([np.where(a == 0, 0.000001, a) for a in rayDirectionsBoxSpace])

        t1 = [np.divide((-b + self.size), a) for a, b in zip(rayDirectionsBoxSpaceNoZeros, rayOriginsBoxSpace)]
        t2 = [np.divide((-b - self.size), a) for a, b in zip(rayDirectionsBoxSpaceNoZeros, rayOriginsBoxSpace)]

        t3 = ([np.where(a < b, a, b) for a, b in np.array(list(zip(t1, t2)))])
        t4 = ([np.where(a > b, a, b) for a, b in np.array(list(zip(t1, t2)))])

        tMin = np.array([np.max(a) for a in t3])
        tMax = np.array([np.min(b) for b in t4])

        tReturn = np.where(np.logical_and(tMin < tMax, tMax > 0), tMin, -1)

        tReturn = np.where(tMin < 0, tMax, tReturn)

        if writeToDict:
            #print("tReturn: ", tReturn)
            tReturnOnlyValidValues = tReturn[tReturn >= 0]
            rayDirectionsBoxSpaceOnlyHit = rayDirectionsBoxSpace[tReturn >= 0]

            t = np.array([(a[0], b[0], a[1], b[1], a[2], b[2]) for a, b in zip(t1, t2)])
            tOnlyValidValues = t[tReturn >= 0]

            tSides = [np.where(a == b)[0] for a, b in zip(tOnlyValidValues, tReturnOnlyValidValues)]
            tSides = [a[0] for a in tSides]

            normalVectors = np.array((self.transformationMatrix[0][:3],  # rechts
                                      - 1 * self.transformationMatrix[0][:3],  # links
                                      self.transformationMatrix[1][:3],  # oben
                                      -1 * self.transformationMatrix[1][:3],  # unten
                                      self.transformationMatrix[2][:3],  # //hinten
                                      -1 * self.transformationMatrix[2][:3]))  # vorne

            # intersecPointsBoxSpace = rayOriginBoxSpace + tReturnOnlyValidValues[...,None]*rayDirectionsBoxSpaceOnlyHit
            # intersecPointsBoxSpace4 = np.insert(np.array(intersecPointsBoxSpace), 3, 1, axis=1)
            # intersecPointsWorldSpace = np.einsum('...ij,...j', self.inverseTransformationMatrix, intersecPointsBoxSpace4)
            # intersecPointsWorldSpace = np.delete(intersecPointsWorldSpace,3,1)

            rayDirectionsOnlyHit = np.array(rayDirections)[tReturn >= 0]
            # print("rayOrigins: ", rayOrigins)
            rayOriginsOnlyHit = rayOrigins[tReturn >= 0]
            intersecPointsWorldSpace = rayOriginsOnlyHit + tReturnOnlyValidValues[..., None] * rayDirectionsOnlyHit

            for i, val in enumerate(intersecPointsWorldSpace):
                self.normalVectorDict[np.array2string(val)] = normalVectors[tSides[i]]

        return tReturn


    def colorsInPoints(self, intersectionPoints, light, camera, currentDepth):
        lightSource = light.origin
        normalVectors = []
        for i in intersectionPoints:
            normalVectors.append(self.normalVectorDict[np.array2string(i)])

        raysToLightSource = np.subtract(lightSource, intersectionPoints)
        raysToLightSource = np.array([ray / np.linalg.norm(ray) for ray in raysToLightSource])
        dotProdsDiffuse = [np.dot(rayToLightSource, normalVector) for rayToLightSource, normalVector in
                           zip(raysToLightSource, normalVectors)]
        # print("dotProdsDiffuse: ", dotProdsDiffuse)
        raysToCamera = np.subtract(camera, intersectionPoints)
        raysToCamera = np.array([ray / np.linalg.norm(ray) for ray in raysToCamera])
        addedRays = raysToLightSource + raysToCamera
        addedRays = np.array([ray / np.linalg.norm(ray) for ray in addedRays])
        dotProdsSpecular = [np.dot(ray, normalVector) for ray, normalVector in zip(addedRays, normalVectors)]
        dotProdsSpecular = np.power(dotProdsSpecular, self.shininess / 4)

        ambient = self.colorAmbient * light.ambient
        ambient = (ambient,) * len(intersectionPoints)
        diffuse = self.colorDiffuse * light.diffuse
        diffuse = (diffuse,) * len(intersectionPoints)
        diffuse = np.array([a * b for a, b in zip(diffuse, dotProdsDiffuse)])
        specular = self.colorSpecular * light.specular
        specular = (specular,) * len(intersectionPoints)
        specular = np.array([a * b for a, b in zip(specular, dotProdsSpecular)])

        colors = ambient + diffuse + specular
        return colors

        # lengths1 = np.square(raysToLightSource[:, 0]) + np.square(raysToLightSource[:, 1]) + np.square(
        #    raysToLightSource[:, 2])
        # lengths1 = np.sqrt(lengths1)
        # lengths = [length1 * length1 * 0.1 * np.linalg.norm(normalVector) for length1, normalVector in zip(lengths1,normalVectors)] # *lengths*0.1 for fsitance to light
        # factors = np.divide(dotProds, lengths)
        # color = (self.color,) * len(intersectionPoints)
        # color = color * factors[:, None]
        # color = np.abs(color)
        # color = color.astype(int)

        # return color

    def colorInPoint(self, intersectionPoint, light, camera):
        lightSource = light.origin
        normalVector = self.normalVectorDict[np.array2string(intersectionPoint)]

        rayToLightSource = np.subtract(lightSource, intersectionPoint)
        rayToLightSource = rayToLightSource / np.linalg.norm(rayToLightSource)
        dotProdDiffuse = np.dot(rayToLightSource, normalVector)

        rayToCamera = np.subtract(camera, intersectionPoint)
        rayToCamera = rayToCamera / np.linalg.norm(rayToCamera)

        addedRay = rayToLightSource + rayToCamera
        addedRay = addedRay / np.linalg.norm(addedRay)

        dotProdSpecular = np.dot(addedRay, normalVector)
        dotProdSpecular = np.power(dotProdSpecular, self.shininess / 4)

        ambient = self.colorAmbient * light.ambient
        diffuse = self.colorDiffuse * light.diffuse * dotProdDiffuse
        specular = self.colorSpecular * light.specular * dotProdSpecular

        color = ambient + diffuse + specular
        return color


    def getNormalVector(self, intersectionPoint):
        return self.normalVectorDict[np.array2string(intersectionPoint)]

    def getReflectedRay(self, intersectionPoint, rayDirection):
        randomVector = np.random.rand(1,3)
        randomVector = randomVector/np.linalg.norm(randomVector)
        randomVector = randomVector * (np.random.rand(1)*2-1)
        # print("isec: ",intersectionPoint, "dir: ", rayDirection)
        rayDirection = rayDirection / np.linalg.norm(rayDirection)
        normalVec = self.normalVectorDict[np.array2string(intersectionPoint)]
        #temp = 2 * np.multiply(np.dot(normalVec, rayDirection), normalVec)
        returnVector = normalVec + randomVector
        returnVector = returnVector/np.linalg.norm(returnVector)
        #print("returnVector", returnVector)
        return returnVector[0]


class Background:
    def __init__(self, color):
        self.color = color


def makeShortestDistancesAndObject(shortestDistancesAndObjects, distancesAndObjects):
    shortestDistances, dummy = zip(*shortestDistancesAndObjects)
    distances, dummy = zip(*distancesAndObjects)

    shortestDistances = np.array(shortestDistances)
    distances = np.array(distances)

    zeros = np.zeros(len(distances))

    shortestDistancesAndObjects = np.where(np.logical_and((zeros <= distances), (distances< shortestDistances))[..., None], distancesAndObjects, shortestDistancesAndObjects)

    return shortestDistancesAndObjects

def makeShadowFactors(distancesToLight, distancesToBody ,shadowFactors):

    distancesToLight = np.array(distancesToLight)
    distancesToBody = np.array(distancesToBody)
    shadowFactors = np.array(shadowFactors)

    zeros = np.zeros(len(distancesToBody))

    shadowFactors = np.where(
        np.logical_and((zeros <= distancesToBody), (distancesToBody < distancesToLight)), shadowFactors - 0.1,
        shadowFactors)

    return shadowFactors


def rayTrace(rayDirections, rayOrigins, scene, light, maxDepth, currentDepth):
    #print("Raytracer: rayDirections:", rayDirections, "rayOrigins:", rayOrigins, "scene:",scene,"light:",light,"maxDepth:",maxDepth,"currenDepth",currentDepth)

    #prepare shortest Distances Array with preloaded values
    shortestDistancesAndObject = np.array(((maxDistance, background),) * len(rayDirections))


    #For every Body check if the Distance to it is the shortest
    for body in scene:
        distancesToBody = body.intersect(rayOrigins, rayDirections, True)
        distancesToBodyAndBody = np.array(list(zip(distancesToBody, np.repeat(body, len(distancesToBody)))))
        shortestDistancesAndObject = makeShortestDistancesAndObject(shortestDistancesAndObject, distancesToBodyAndBody)

    #calcute Intersectionpoints from distances
    shortestDistances, shortestDistancesObjects = zip(*shortestDistancesAndObject)
    shortestDistances = np.array(shortestDistances)
    intersectionPoints = rayOrigins + rayDirections * shortestDistances[:, None]
    shortestDistancesIntersectionsDirectionsObjects = np.array(list(zip(shortestDistances,intersectionPoints, rayDirections, shortestDistancesObjects)))

    #preparations for shadows
    displacedIntersectionPoints = np.array(
        [point + 0.001 * body.getNormalVector(point) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects])
    raysFromLightToDisplacedPoint = np.subtract(displacedIntersectionPoints, light.origin)
    distancesToLight = np.array([np.linalg.norm(ray) for ray in raysFromLightToDisplacedPoint])

    #shortestDistanceForShadowsAndFactor = np.array(list(zip(distancesToLight, np.ones(len(distancesToLight)))))

    shadowFactors = np.ones(len(distancesToLight))

    for body in scene:
        for i in range(4):
            random = np.random.rand(1,2)
            random = (random[0][0]*2,random[0][1]*2)
            random = (random[0]-1, random[1]-1)
            point = [light.origin[0]+light.halfLength*random[0],light.origin[1],light.origin[2]+light.halfLength*random[1]]
            print(point)
            lightVerticeAsArray = (point,) * len(rayDirections)
            lightVerticeAsArray = np.array(lightVerticeAsArray)

            raysFromLightToDisplacedPoint = np.subtract(displacedIntersectionPoints, point)
            distancesToBody = body.intersect(lightVerticeAsArray, raysFromLightToDisplacedPoint, False)
            distancesToLight = np.array([np.linalg.norm(ray) for ray in raysFromLightToDisplacedPoint])

            shadowFactors = makeShadowFactors(distancesToLight, distancesToBody, shadowFactors)

    factorsArray = shadowFactors
    #dummy, factorsArray = zip(*shortestDistanceForShadowsAndFactor )


    if 1:
        #print("Old")
        allIndices = []
        allBodyColors = []

        for body in scene:
            indicesOfPixels = np.where(shortestDistancesIntersectionsDirectionsObjects[:, 3] == body)[0]
            sDAIPAOofBody = shortestDistancesIntersectionsDirectionsObjects[shortestDistancesIntersectionsDirectionsObjects[:, 3] == body]

            dummy, bodyIntersectionPoints, dummy, dummy = zip(*sDAIPAOofBody)
            bodyColors = body.colorsInPoints(bodyIntersectionPoints, light, camera, currentDepth)
            allIndices.extend(indicesOfPixels)
            allBodyColors.extend(bodyColors)

        allIndices = np.array(allIndices)
        allBodyColors = np.array(allBodyColors)
        sorter = np.argsort(allIndices)

        colors = allBodyColors[sorter]

    else:
        #print("New")
        colors = [body.colorInPoint(point, light, camera) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects]

    colors = np.array([color * factorsArray[i] for i, color in enumerate(colors)])

    reflectedRays = [body.getReflectedRay(point,ray) for dummmy, point, ray, body in shortestDistancesIntersectionsDirectionsObjects]
    reflectedRays = [ray/np.linalg.norm(ray) for ray in reflectedRays]
    reflectedRays = [ray.tolist() for ray in reflectedRays]

    reflectionFactors = np.array([body.reflection for dummy, dummy, dummy, body in shortestDistancesIntersectionsDirectionsObjects])
    #for i in reflectionFactors:
    #    print("factor:",i)
    #print("reflectedRay:",reflectedRays)

    #return colors

    if currentDepth < maxDepth:
        extraColors = rayTrace(reflectedRays, displacedIntersectionPoints, scene, light, maxDepth, currentDepth+1)
        #print("extraColorsBeforeFactoring:",extraColors)
        extraColors = reflectionFactors[:, None] * extraColors
        #print("extraColorsAfterFactoring:",extraColors)
        #for i in extraColors:
        #    print(i)
        fullColors = colors + extraColors
        #print("fullcolors:", fullColors)
        return fullColors
    else:
        return colors


hinten = Plane((0, 0, -1), 10, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (0, 0, 0), 10000, 0.01)
rechts = Plane((-1, 0, 0), 10, (0.1, 0, 0), (0.7, 0, 0), (0, 0, 0), 10000, 0.01)
unten = Plane((0, 1, 0), 0, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (0, 0, 0), 10000, 0.01)
oben = Plane((0, -1, 0), 10, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (0, 0, 0), 10000, 0.01)
links = Plane((1, 0, 0), 0, (0.1, 0, 0.1), (0.7, 0, 0.7), (0, 0, 0), 10000, 0.01)
behind = Plane((1, 0, 0), 10, (0.1, 0, 0.1), (0.7, 0, 0.7), (0, 0, 0), 10000, 0.01)
cube1 = Cuboid((1.5, 1.5, 1.5), (2.5, 1.5, 4), 0, (0, 0.1, 0.1), (0, 0.7, 0.7), (1, 1, 1), 100, 0.2)
cube2 = Cuboid((1.5, 3, 1.5), (7, 3, 5), 45, (0.1, 0.1, 0), (0.7, 0.7, 0), (1, 1, 1), 100, 0.5)

scene = [rechts, hinten, links, unten, oben, cube1, cube2]

xResolution = 100
yResolution = 100
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
light = Light((5, 9, 5), (1, 1, 1), (1, 1, 1), (1, 1, 1), 0.5)
background = Background((0, 0, 0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0], xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1], yResolution)

pixelCoords = [(x, y, 0) for y in pixelCoordsY for x in pixelCoordsX]



pixelRays = np.subtract(pixelCoords, camera)
pixelRays = np.array([pixelRay / np.linalg.norm(pixelRay) for pixelRay in pixelRays])

cameraCoordsArray = (camera,) * len(pixelCoords)
cameraCoordsArray = np.array(cameraCoordsArray)

starttime = time.time()
colors = rayTrace(pixelRays, cameraCoordsArray, scene, light, 1, 0)
timeTook = time.time() - starttime
print("It took:", timeTook, "s")  # 50x50 = 9.2s 100x100 = 36.9


colors = np.reshape(colors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()


