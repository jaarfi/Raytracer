import numpy as np

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

    rayDirections = rays.directions
    rayOrigins = rays.origins


    maxRange = np.array((1e19,)*len(rayOrigins))
    interPoints = np.array(([0,0,0],)*len(rayOrigins))
    normalVectors = np.array(([0,0,0],)*len(rayOrigins))
    colors = np.array(([0,0,0],)*len(rayOrigins))
    bodies = np.zeros(len(rayOrigins))
    hitInformations = IntersecPointInformations(maxRange, interPoints, normalVectors, colors,bodies)

    time1 = time.time()
    for body in scene:
        print(body)
        intersectionInformations = body.intersect(rayOrigins, rayDirections, True)
        hitInformations = getShortestDistancesInformations(hitInformations, intersectionInformations)

    time2 = time.time() - time1
    print("intersecting took", time2)

    colors = rays.getColors(hitInformations, scene, light)

    return colors
