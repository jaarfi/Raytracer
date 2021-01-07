import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math


class Light:
    def __init__(self, origin, ambient, diffuse, specular, halfLength, intensity):
        self.origin = np.array(origin)
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.halfLength = halfLength

        tempArray = np.array([[halfLength,0,halfLength],[-halfLength,0,halfLength],[halfLength,0,-halfLength],[-halfLength,0,-halfLength]])
        self.vertices = [origin+tempArray][0]
        self.intensity = intensity

class ReturnVal:
    def __init__(self, dist, point, normalVector):
        self.dist = dist
        self.point = point
        self.normalVector = normalVector

class Plane:
    def __init__(self, normalVector, distanceToOrigin, color, shininess, albedo, kd, ks):
        self.normalVector = normalVector / np.linalg.norm(normalVector)
        self.distanceToOrigin = distanceToOrigin
        self.color = color
        self.shininess = shininess
        self.albedo = albedo
        self.kd = kd
        self.ks = ks

    def intersect(self, rayOrigins, rayDirections, writeToDict):
        rayDirections = np.array([ray / np.linalg.norm(ray) for ray in rayDirections])
        a = np.array([-(np.dot(ray, self.normalVector) + self.distanceToOrigin) for ray in rayOrigins])
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b == 0, -1, b)

        return a / b

    def colorInPointLambert(self, intersectionPoint, light):
        normalVector = self.normalVector
        rayToLightSource = np.subtract(light.origin, intersectionPoint)
        lightIntensity = light.intensity / (4 * np.pi * np.linalg.norm(rayToLightSource))
        cos = np.dot(rayToLightSource, normalVector) / (np.linalg.norm(rayToLightSource) * np.linalg.norm(normalVector))
        if cos < 0:
            return [0, 0, 0]
        color = np.array(self.color)
        color = color * cos
        color = color * lightIntensity * self.albedo / np.pi
        return color

    def colorInPointPhong(self, intersectionPoint, light, camera):
        normalVector = self.normalVector

        rayToLightSource = np.subtract(light.origin, intersectionPoint)
        lightIntensity = light.intensity / (4 * np.pi * np.linalg.norm(rayToLightSource))
        cosDiffuse = np.dot(rayToLightSource, normalVector) / (np.linalg.norm(rayToLightSource) * np.linalg.norm(normalVector))
        if cosDiffuse <= 0:
            return [0, 0, 0]
        diffuse = np.array(self.color)
        diffuse = diffuse * cosDiffuse
        diffuse = diffuse * lightIntensity * self.albedo / np.pi

        rayFromCamera = np.subtract(intersectionPoint, camera)

        optimalReflection = reflect(rayFromCamera, normalVector)
        cosSpecular = np.dot(optimalReflection, rayFromCamera) / (
                    np.linalg.norm(optimalReflection) * np.linalg.norm(rayFromCamera))
        if cosSpecular <= 0:
            return diffuse
        specular = np.ones(3)
        specular = lightIntensity * np.power(cosSpecular, self.shininess)

        color = diffuse * self.kd + specular * self.ks
        return color

    def getNormalVector(self, intersectionPoint):
        return self.normalVector

class Cuboid:
    def __init__(self, size, center, rotation, color, shininess, albedo, kd, ks):
        self.size = size
        self.rotation = np.radians(rotation)
        self.center = center

        self.color = color
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
        self.albedo = albedo
        self.kd = kd
        self.ks = ks


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
            rayOriginsOnlyHit = np.array(rayOrigins)[tReturn >= 0]
            intersecPointsWorldSpace = rayOriginsOnlyHit + tReturnOnlyValidValues[..., None] * rayDirectionsOnlyHit

            for i, val in enumerate(intersecPointsWorldSpace):
                self.normalVectorDict[np.array2string(val)] = normalVectors[tSides[i]]

        return tReturn

    def colorInPointLambert(self, intersectionPoint, light):
        normalVector = self.normalVectorDict[np.array2string(intersectionPoint)]
        rayToLightSource = np.subtract(light.origin, intersectionPoint)
        cos = np.dot(rayToLightSource, normalVector) / (np.linalg.norm(rayToLightSource)*np.linalg.norm(normalVector))
        if cos < 0:
            return [0,0,0]
        color = np.array(self.color)
        color = color * cos
        color = color * light.intensity * self.albedo / np.pi
        color = color / (4 * np.pi * np.linalg.norm(rayToLightSource))
        return color

    def colorInPointPhong(self, intersectionPoint, light, camera):
        normalVector = self.normalVectorDict[np.array2string(intersectionPoint)]

        rayToLightSource = np.subtract(light.origin, intersectionPoint)
        lightIntensity = light.intensity / (4 * np.pi * np.linalg.norm(rayToLightSource))
        cosDiffuse = np.dot(rayToLightSource, normalVector) / (np.linalg.norm(rayToLightSource) * np.linalg.norm(normalVector))
        if cosDiffuse <= 0:
            return [0, 0, 0]
        diffuse = np.array(self.color)
        diffuse = diffuse * cosDiffuse
        diffuse = diffuse * lightIntensity * self.albedo / np.pi

        rayFromCamera = np.subtract(intersectionPoint, camera)

        optimalReflection = reflect(rayFromCamera, normalVector)
        cosSpecular = np.dot(optimalReflection, rayFromCamera) / (
                    np.linalg.norm(optimalReflection) * np.linalg.norm(rayFromCamera))
        if cosSpecular <= 0:
            return diffuse
        specular = lightIntensity * np.power(cosSpecular, self.shininess)

        color = diffuse * self.kd + specular * self.ks
        return color

    def getNormalVector(self, intersectionPoint):
        return self.normalVectorDict[np.array2string(intersectionPoint)]


class Background:
    def __init__(self, color):
        self.color = color

    def colorInPoint(self, intersectionPoint, light, camera):
        return self.color

    def getNormalVector(self, intersectionPoint):
        return np.array([0,0,0])

    def colorInPointLambert(self, intersectionPoint, light):
        return np.array([0,0,0])

def reflect(rayIncoming, normalVector):
    rayOutgoing = rayIncoming - 2 * np.dot(rayIncoming, normalVector) * normalVector
    return rayOutgoing


def makeShortestDistancesAndObject(shortestDistancesAndObjects, distancesAndObjects):
    shortestDistances, dummy = zip(*shortestDistancesAndObjects)
    distances, dummy = zip(*distancesAndObjects)

    shortestDistances = np.array(shortestDistances)
    distances = np.array(distances)

    zeros = np.zeros(len(distances))

    shortestDistancesAndObjects = np.where(np.logical_and((zeros <= distances), (distances< shortestDistances))[..., None], distancesAndObjects, shortestDistancesAndObjects)

    return shortestDistancesAndObjects


def randomVectors(normalVectors):
    randomVectors = np.random.rand(len(normalVectors),3)
    randomVectors = np.array([vec / np.linalg.norm(vec) for vec in randomVectors])
    randomVectors = np.array([vec * (np.random.rand(1) * 2 - 1) for vec in randomVectors])

    randomVectors = randomVectors + normalVectors
    #print("randomVectors:", randomVectors)
    return randomVectors


def makeShadowFactors(distancesToLight, distancesToBody ,shadowFactors):

    distancesToLight = np.array(distancesToLight)
    distancesToBody = np.array(distancesToBody)
    shadowFactors = np.array(shadowFactors)

    zeros = np.zeros(len(distancesToBody))

    shadowFactors = np.where(
        np.logical_and((zeros <= distancesToBody), (distancesToBody < distancesToLight)), shadowFactors - 1,
        shadowFactors)

    return shadowFactors

def rayTrace(rayDirections, rayOrigins, scene, light, maxDepth, currentDepth):
    shortestDistancesAndObject = np.array(((maxDistance, background),) * len(rayDirections))

    time1 = time.time()
    for body in scene:
        distancesToBody = body.intersect(rayOrigins, rayDirections, True)
        distancesToBodyAndBody = np.array(list(zip(distancesToBody, np.repeat(body, len(distancesToBody)))))
        shortestDistancesAndObject = makeShortestDistancesAndObject(shortestDistancesAndObject, distancesToBodyAndBody)

    time2 = time.time() - time1
    print("intersecting took", time2)

    shortestDistances, shortestDistancesObjects = zip(*shortestDistancesAndObject)
    shortestDistances = np.array(shortestDistances)
    intersectionPoints = rayOrigins + rayDirections * shortestDistances[:, None]
    shortestDistancesIntersectionsDirectionsObjects = np.array(list(zip(shortestDistances,intersectionPoints, rayDirections, shortestDistancesObjects)))
    displacedIntersectionPoints = np.array([point + 0.001 * body.getNormalVector(point) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects])

    normalVectors = [body.getNormalVector(point) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects]

    displacedDirectionsObjectsNormals = np.array(list(zip(displacedIntersectionPoints, rayDirections, shortestDistancesObjects, normalVectors)))
    #raysFromLightToDisplacedPoint = np.subtract(displacedIntersectionPoints, light.origin)
    #distancesToLight = np.array([np.linalg.norm(ray) for ray in raysFromLightToDisplacedPoint])

    shadowFactors = np.ones(len(rayOrigins))

    time1 = time.time()
    for body in scene:
        for point in [light.origin]:
            random = np.random.rand(1, 2)
            random = (random[0][0] * 2, random[0][1] * 2)
            random = (random[0] - 1, random[1] - 1)
            #point = [light.origin[0] + light.halfLength * random[0], light.origin[1], light.origin[2] + light.halfLength * random[1]]
            lightVerticeAsArray = (point,) * len(rayDirections)
            lightVerticeAsArray = np.array(lightVerticeAsArray)

            raysFromLightToDisplacedPoint = np.subtract(displacedIntersectionPoints, point)
            distancesToBody = body.intersect(lightVerticeAsArray, raysFromLightToDisplacedPoint, False)
            distancesToLight = np.array([np.linalg.norm(ray) for ray in raysFromLightToDisplacedPoint])

            shadowFactors = makeShadowFactors(distancesToLight, distancesToBody, shadowFactors)

    factorsArray = np.array(shadowFactors)
    print(factorsArray)
    time2 = time.time() - time1
    print("shading took", time2)

    time1 = time.time()
    colors = np.array([body.colorInPointLambert(point, light) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects])
    time2 = time.time() - time1
    print("coloring took", time2)
    #colors = np.array([body.colorInPointPhong(point, light, camera) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects])

    time1 = time.time()
    colors = colors + [monteCarlo(ray, point, body, normal, light, 10, scene) for point, ray, body, normal in displacedDirectionsObjectsNormals]
    time2 = time.time() - time1
    print("Montecarlo took", time2)
    #colors = np.array([color * factorsArray[i] for i, color in enumerate(colors)])


    return colors

def monteCarlo(rayDirection, intersectionPoint, body, normalVector, light, n, scene):
    rayToLight = np.subtract(light.origin, intersectionPoint)
    normalVectorArray = (normalVector,) * n
    reflectedVectors = randomVectors(normalVectorArray)
    #print("reflec:", reflectedVectors)
    shortestDistancesAndObject = np.array(((maxDistance, background),) * n)
    intersectionPointArray = (intersectionPoint,) *n

    time1 = time.time()
    for body in scene:
        distancesToBody = body.intersect(intersectionPointArray, reflectedVectors, True)
        distancesToBodyAndBody = np.array(list(zip(distancesToBody, np.repeat(body, len(distancesToBody)))))
        shortestDistancesAndObject = makeShortestDistancesAndObject(shortestDistancesAndObject, distancesToBodyAndBody)

    time2 = time.time() - time1
    #print("Montecarlo intersecting took", time2)

    shortestDistances, shortestDistancesObjects = zip(*shortestDistancesAndObject)
    shortestDistances = np.array(shortestDistances)
    intersectionPoints = intersectionPoint + reflectedVectors * shortestDistances[:, None]
    shortestDistancesIntersectionsDirectionsObjects = np.array(
        list(zip(shortestDistances, intersectionPoints, reflectedVectors, shortestDistancesObjects)))


    colors = np.array([body.colorInPointLambert(point, light) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects])

    cosPaths = [np.dot(reflectedVector, rayToLight)/(np.linalg.norm(reflectedVectors)*np.linalg.norm(rayToLight)) for reflectedVector in reflectedVectors]

    colors = [color * cos for color, cos in zip(colors, cosPaths)]

    colorSum = np.sum(colors, axis=0)
    colorSum = colorSum / n
    colorSum = colorSum * body.albedo * 2
    #print(colors)
    return colorSum

def monteCarloArray(rayDirection, intersectionPoint, body, normalVector, light, n, scene):
    rayToLight = np.subtract(light.origin, intersectionPoint)
    normalVectorArray = (normalVector,) * n
    reflectedVectors = randomVectors(normalVectorArray)
    #print("reflec:", reflectedVectors)
    shortestDistancesAndObject = np.array(((maxDistance, background),) * n)
    intersectionPointArray = (intersectionPoint,) *n

    time1 = time.time()
    for body in scene:
        distancesToBody = body.intersect(intersectionPointArray, reflectedVectors, True)
        distancesToBodyAndBody = np.array(list(zip(distancesToBody, np.repeat(body, len(distancesToBody)))))
        shortestDistancesAndObject = makeShortestDistancesAndObject(shortestDistancesAndObject, distancesToBodyAndBody)

    time2 = time.time() - time1
    #print("Montecarlo intersecting took", time2)

    shortestDistances, shortestDistancesObjects = zip(*shortestDistancesAndObject)
    shortestDistances = np.array(shortestDistances)
    intersectionPoints = intersectionPoint + reflectedVectors * shortestDistances[:, None]
    shortestDistancesIntersectionsDirectionsObjects = np.array(
        list(zip(shortestDistances, intersectionPoints, reflectedVectors, shortestDistancesObjects)))


    colors = np.array([body.colorInPointLambert(point, light) for dummy, point, dummy, body in shortestDistancesIntersectionsDirectionsObjects])

    cosPaths = [np.dot(reflectedVector, normalVector)/(np.linalg.norm(reflectedVectors)*np.linalg.norm(normalVector)) for reflectedVector in reflectedVectors]

    colors = [color * cos for color, cos in zip(colors, cosPaths)]

    colorSum = np.sum(colors, axis=0)
    colorSum = colorSum / n
    colorSum = colorSum * body.albedo * 2
    #print(colors)
    return colorSum


hinten = Plane((0, 0, -1), 10, (0.7, 0.7, 0.7), 10000, 0.18, 0.88, 0.12)
rechts = Plane((-1, 0, 0), 10, (0, 0, 0.7), 10000,  0.18, 0.88,  0.12)
unten = Plane((0, 1, 0), 0, (0.7, 0.7, 0.7), 10000,  0.18, 0.88,  0.12)
oben = Plane((0, -1, 0), 10, (0.7, 0.7, 0.7), 10000,  0.18, 0.88,  0.12)
links = Plane((1, 0, 0), 0, (0.7, 0, 0), 10000,  0.18, 0.88,  0.12)
behind = Plane((1, 0, 0), 10, (0.7, 0, 0.7), 10000,  0.18, 0.88,  0.12)
cube1 = Cuboid((1.5, 1.5, 1.5), (2.5, 1.5, 4), 0, (0.7, 0.7, 0), 100, 0.2, 0.88,  0.12)
cube2 = Cuboid((1.5, 3, 1.5), (7, 3, 5), 45, (0.7, 0.7, 0.7), 100, 0.5, 0.88,  0.12)

scene = [rechts, hinten, links, unten, oben, cube1, cube2]

xResolution = 100
yResolution = 100
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
intensity = 1200
light = Light((5, 9, 5), (1, 1, 1), (1, 1, 1), (1, 1, 1), 0.5, intensity)
background = Background((0, 0, 0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0], xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1], yResolution)

pixelCoords = [(x, y, 0) for y in pixelCoordsY for x in pixelCoordsX]



pixelRays = np.subtract(pixelCoords, camera)
pixelRays = np.array([pixelRay / np.linalg.norm(pixelRay) for pixelRay in pixelRays])

cameraCoordsArray = (camera,) * len(pixelCoords)
cameraCoordsArray = np.array(cameraCoordsArray)

starttime = time.time()
colors = rayTrace(pixelRays, cameraCoordsArray, scene, light, 2, 0)

timeTook = time.time() - starttime
print("It took:", timeTook, "s")  # 50x50 = 9.2s 100x100 = 36.9


colors = np.reshape(colors, (xResolution, yResolution, 3))

plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()


