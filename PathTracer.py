import matplotlib.pyplot as plt
from CustomClass.Plane import *
from CustomClass.Cuboid import *
from CustomClass.Background import *
from CustomClass.Light import *
from CustomClass.Rays import *
from CustomClass.Surfaces import *


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

xResolution = 100
yResolution = 100
maxDistance = 1e6

camera = (5, 5, -5)
screen = ((0, 0), (10, 10))
background = Background((0, 0, 0))

pixelCoordsX = np.linspace(screen[0][0], screen[1][0], xResolution)
pixelCoordsY = np.linspace(screen[0][1], screen[1][1], yResolution)
zeros = np.zeros(len(pixelCoordsX))

pixelCoords = np.array(np.meshgrid(pixelCoordsX, pixelCoordsY, 0)).T.reshape(-1,3)


pixelRays = np.subtract(pixelCoords, camera)
pixelRayLengths = np.apply_along_axis(np.linalg.norm, 0, pixelRays)
pixelRays = pixelRays / pixelRayLengths

cameraCoordsArray = (camera,) * len(pixelCoords)
cameraCoordsArray = np.array(cameraCoordsArray)

starttime = time.time()

samples = 1
maxDepth = 4
rays = Rays(cameraCoordsArray, pixelRays, maxDepth, 0)

allColors = rays.getColors(scene)
for i in range(samples-1):
    colors = rays.getColors(scene)
    allColors = allColors + colors
allColors = np.array(allColors)
allColors = allColors/samples

timeTook = time.time() - starttime
print("It took:", timeTook, "s")  # 1sample, 1ray, 4depth, 100r = 12.27s


colors = np.reshape(allColors, (xResolution, yResolution, 3))
plt.imshow(colors)
plt.gca().invert_yaxis()
plt.show()

