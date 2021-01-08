import matplotlib.pyplot as plt
import time
from CustomClass.Plane import *
from CustomClass.Cuboid import *
from CustomClass.Background import *
from CustomClass.Light import *
from CustomClass.Rays import *
from CustomClass.Surfaces import *






def getRaysColors(rays, scene, light):
    pass

def materialDiffuse(scene, intersectionPoints, light):
    pass


def getShortestDistancesInformations(shortestIntersectionInformations, bodyIntersectionInformations):
    shortestDistances = shortestIntersectionInformations.distances

    bodyDistances = bodyIntersectionInformations.distances

    shortestIntersectionInformationsZipped = informationToZipped(shortestIntersectionInformations)
    bodyIntersectionInformationsZipped = informationToZipped(bodyIntersectionInformations)

    zeros = np.zeros(len(shortestDistances))


    finalZip = np.where(np.logical_and((zeros <= bodyDistances), (bodyDistances< shortestDistances))[..., None], bodyIntersectionInformationsZipped, shortestIntersectionInformationsZipped)

    dist, insec, norm,  col, bodies = zip(*finalZip)

    #print(dist)
    returnInformation = IntersecPointInformations(dist,insec,norm, col, bodies)


    return returnInformation

def informationToZipped(intersectionPointInformations):
    dist = intersectionPointInformations.distances
    insec = intersectionPointInformations.points
    normals = intersectionPointInformations.normalVectors
    disp = intersectionPointInformations.displacedPoints
    bodies = intersectionPointInformations.bodies

    return np.array(list(zip(dist, insec, normals, disp, bodies)))


def rayTrace(rays, scene, light):



    colors = rays.getColors(scene, light)

    return colors

matt_blue = Matt((0,0,0.7))
matt_grey = Matt((0.7,0.7,0.7))
matt_red = Matt((0.7,0,0))
matt_yellow = Matt((0.7,0.7,0))
radiant_white = Radiant((15,15,15))

hinten = Plane( matt_grey, (0, 0, -1), 10, 10000, 0.01)
rechts = Plane(matt_red, (-1, 0, 0), 10, 10000, 0.01)
unten = Plane(matt_grey, (0, 1, 0), 0, 10000, 0.01)
oben = Plane(matt_grey, (0, -1, 0), 10, 10000, 0.01)
links = Plane(matt_blue, (1, 0, 0), 0, 10000, 0.01)
light = AxisAlignedSquare(radiant_white, (5,9.99,3), 3, 0, 0)
behind = Plane(matt_grey, (1, 0, 0), 10, 10000, 0.01)
cube1 = Cuboid(matt_yellow, (1.5, 1.5, 1.5), (5, 1.5, 3), 0, 100, 0.2)
#cube2 = Cuboid((1.5, 3, 1.5), (7, 3, 5), 45, (0.1, 0.1, 0.1), (0.7, 0.7, 0.7), (1, 1, 1), 100, 0.5)

scene = [rechts, hinten, links, unten, oben, light, behind, cube1]

xResolution = 200
yResolution = 200
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
light = Light((5, 9, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1), 0.5, 1000)
background = Background((0, 0, 0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0], xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1], yResolution)

pixelCoords = [(x, y, 0) for y in pixelCoordsY for x in pixelCoordsX]



pixelRays = np.subtract(pixelCoords, camera)
pixelRays = np.array([pixelRay / np.linalg.norm(pixelRay) for pixelRay in pixelRays])

cameraCoordsArray = (camera,) * len(pixelCoords)
cameraCoordsArray = np.array(cameraCoordsArray)

starttime = time.time()

samples = 20
maxDepth = 2
rays = Rays(cameraCoordsArray, pixelRays, maxDepth, 0)

allColors = rayTrace(rays, scene, light)
for i in range(samples-1):
    colors = rayTrace(rays,  scene, light)
    allColors = allColors + colors
allColors = np.array(allColors)
allColors = allColors/samples

timeTook = time.time() - starttime
print("It took:", timeTook, "s")  # res.depth.nrrays.samples 100x2x20x4 = 562s 100x2x5x4 = 146s 100x2x20x1 = 143s


colors = np.reshape(allColors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()


