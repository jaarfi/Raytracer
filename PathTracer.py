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
        self.intensity = intensity

        tempArray = np.array([[halfLength,0,halfLength],[-halfLength,0,halfLength],[halfLength,0,-halfLength],[-halfLength,0,-halfLength]])
        self.vertices = [origin+tempArray][0]

class Rays:
    def __init__(self, origins, directions, depth, n, reflections):
        self.origins = origins  # the point where the ray comes from
        self.directions = directions  # direction of the ray
        self.depth = depth  # ray_depth is the number of the refrections + transmissions/refractions, starting at zero for camera rays
        self.n = n  # ray_n is the index of refraction of the media in which the ray is travelling
        self.reflections = reflections

def getRaysColors(rays, scene, light):
    pass

def materialDiffuse(scene, intersectionPoints, light):
    pass

def getDiffuseLambert(intersectionPoints, light, normalVectors, colors):

        print("pints",intersectionPoints)
        raysToLightSource = np.subtract(light.origin, intersectionPoints)
        cos = np.array([np.dot(ray, normalVector) / (np.linalg.norm(ray)*np.linalg.norm(normalVector)) for ray,normalVector in zip(raysToLightSource,normalVectors)])
        colors = np.array([np.multiply(color, a) for color, a in zip(colors,cos)])
        #print("1",colors)
        colors = colors * light.intensity * 0.18 / np.pi
        #print("2",colors)
        colors = np.array([color / (4 * np.pi * np.linalg.norm(ray)) for color, ray in zip(colors,raysToLightSource)])
        #print("3",colors)
        return colors

class IntersecPointInformations:
    def __init__(self, distances, points, normalVectors, colors):
        self.distances = np.array(distances)
        self.points = np.array(points)
        self.normalVectors = np.array(normalVectors)
        #print("normals", self.normalVectors)
        self.displacedPoints = np.add(self.points, self.normalVectors * 0.01)
        self.colors = colors



class ReturnVal:
    def __init__(self, dist, point, normalVector):
        self.dist = dist
        self.point = point
        self.normalVector = normalVector

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
        rayDirections = np.array([ray / np.linalg.norm(ray) for ray in rayDirections])
        a = np.array([-(np.dot(ray, self.normalVector) + self.distanceToOrigin) for ray in rayOrigins])
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b == 0, -1, b)
        normalVectorArray = (self.normalVector,) * len(rayOrigins)
        #print("normalVectorArray:", normalVectorArray)
        colorsArray = (self.colorDiffuse, ) * len(rayOrigins)

        distances = a/b
        intersecPoints = [distance * direction + origin for distance, direction, origin in zip(distances,rayDirections,rayOrigins)]
        print("distances:", distances)
        print("insecpoints:", intersecPoints)
        infos = IntersecPointInformations(a/b, intersecPoints, normalVectorArray, colorsArray)
        return infos

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


        #print("tReturn: ", tReturn)

        t = np.array([(a[0], b[0], a[1], b[1], a[2], b[2]) for a, b in zip(t1, t2)])

        t = [np.append(a,-1) for a in t]

        #print(tReturn)

        tSides = [np.where(a == b)[0] for a, b in zip(t, tReturn)]

        #print(tSides)
        tSides = [a[0] for a in tSides]

        normalVectors = np.array((self.transformationMatrix[0][:3],  # rechts
                                  - 1 * self.transformationMatrix[0][:3],  # links
                                  self.transformationMatrix[1][:3],  # oben
                                  -1 * self.transformationMatrix[1][:3],  # unten
                                  self.transformationMatrix[2][:3],  # //hinten
                                  -1 * self.transformationMatrix[2][:3],
                                 [0,0,0]))  # vorne


        intersecPointsWorldSpace = rayOrigins + tReturn[..., None] * rayDirections

        normalVectorArray = [normalVectors[tSides]][0]

        colorsArray = (self.colorDiffuse, ) * len(rayOrigins)
        infos = IntersecPointInformations(tReturn, intersecPointsWorldSpace, normalVectorArray, colorsArray)
        return infos

class Background:
    def __init__(self, color):
        self.color = color

    def colorInPoint(self, intersectionPoint, light, camera):
        return self.color

    def getNormalVector(self, intersectionPoint):
        return np.array([0,0,0])

def getShortestDistancesInformations(shortestIntersectionInformations, bodyIntersectionInformations):
    shortestDistances = shortestIntersectionInformations.distances
    shortestIntersections = shortestIntersectionInformations.points
    shortestNormals = shortestIntersectionInformations.normalVectors
    shortestDisplaced = shortestIntersectionInformations.displacedPoints
    shortestColors = shortestIntersectionInformations.colors

    bodyDistances = bodyIntersectionInformations.distances
    bodyIntersections = bodyIntersectionInformations.points
    bodyNormals = bodyIntersectionInformations.normalVectors
    bodyDisplaced = bodyIntersectionInformations.displacedPoints
    bodyColors = bodyIntersectionInformations.colors

    shortestIntersectionInformationsZipped = np.array(list(zip(shortestDistances, shortestIntersections, shortestNormals, shortestDisplaced, shortestColors)))
    bodyIntersectionInformationsZipped = np.array(list(zip(bodyDistances, bodyIntersections, bodyNormals, bodyDisplaced, bodyColors)))

    zeros = np.zeros(len(shortestDistances))


    finalZip = np.where(np.logical_and((zeros <= bodyDistances), (bodyDistances< shortestDistances))[..., None], bodyIntersectionInformationsZipped, shortestIntersectionInformationsZipped)

    dist, insec, norm, disp, col = zip(*finalZip)

    #print(dist)
    returnInformation = IntersecPointInformations(dist,insec,norm,col)


    return returnInformation

def rayTrace(rayDirections, rayOrigins, scene, light, maxDepth, currentDepth):

    maxRange = np.array((1e19,)*len(rayOrigins))
    interPoints = np.array(([0,0,0],)*len(rayOrigins))
    normalVectors = np.array(([0,0,0],)*len(rayOrigins))
    colors = np.array(([0,0,0],)*len(rayOrigins))
    hitInformations = IntersecPointInformations(maxRange, interPoints, normalVectors, colors)

    time1 = time.time()
    for body in scene:
        intersectionInformations = body.intersect(rayOrigins, rayDirections, True)
        hitInformations = getShortestDistancesInformations(hitInformations, intersectionInformations)

    time2 = time.time() - time1
    print("intersecting took", time2)

    displacedIntersectionPoints = hitInformations.displacedPoints
    distances = hitInformations.distances
    intersectionPoints = hitInformations.points
    print(distances[distances < 0])
    normalVectors = hitInformations.normalVectors
    colors = hitInformations.colors

    time1 = time.time()
    colors = getDiffuseLambert(intersectionPoints, light, normalVectors, colors)
    time2 = time.time() - time1
    print("coloring took", time2)
    return colors

hinten = Plane((0, 0, -1), 10, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (0, 0, 0), 10000, 0.01)
rechts = Plane((-1, 0, 0), 10, (0, 0, 0.1), (0, 0, 0.5), (0, 0, 0), 10000, 0.01)
unten = Plane((0, 1, 0), 0, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (0, 0, 0), 10000, 0.01)
oben = Plane((0, -1, 0), 10, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (0, 0, 0), 10000, 0.01)
links = Plane((1, 0, 0), 0, (0.1, 0, 0), (0.6, 0, 0), (0, 0, 0), 10000, 0.01)
behind = Plane((1, 0, 0), 10, (0.1, 0, 0.1), (0.4, 0, 0.4), (0, 0, 0), 10000, 0.01)
cube1 = Cuboid((1.5, 1.5, 1.5), (2.5, 1.5, 4), 0, (0.1, 0.1, 0), (0.7, 0.7, 0), (1, 1, 1), 100, 0.2)
cube2 = Cuboid((1.5, 3, 1.5), (7, 3, 5), 45, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (1, 1, 1), 100, 0.5)

scene = [rechts, hinten, links, unten, oben, cube1, cube2]

xResolution = 100
yResolution = 100
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
light = Light((5, 9, 5), (1, 1, 1), (1, 1, 1), (1, 1, 1), 0.5, 1000)
background = Background((0, 0, 0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0], xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1], yResolution)

pixelCoords = [(x, y, 0) for y in pixelCoordsY for x in pixelCoordsX]



pixelRays = np.subtract(pixelCoords, camera)
pixelRays = np.array([pixelRay / np.linalg.norm(pixelRay) for pixelRay in pixelRays])

cameraCoordsArray = (camera,) * len(pixelCoords)
cameraCoordsArray = np.array(cameraCoordsArray)

starttime = time.time()

samples = 1

allColors = rayTrace(pixelRays, cameraCoordsArray, scene, light, 2, 0)
for i in range(samples-1):
    colors = rayTrace(pixelRays, cameraCoordsArray, scene, light, 2, 0)
    allColors = allColors + colors
allColors = np.array(allColors)
allColors = allColors/samples

timeTook = time.time() - starttime
print("It took:", timeTook, "s")  # 50x50 = 9.2s 100x100 = 36.9


colors = np.reshape(allColors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()


