import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np
import typing
import time
import math
from CustomClass.IntersecPointInformations import *
from CustomClass.Plane import *
from CustomClass.Cuboid import *
from CustomClass.Background import *
from CustomClass.Light import *






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


