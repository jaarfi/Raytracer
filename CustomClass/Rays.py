import numpy as np
import time
from .IntersecPointInformations import IntersecPointInformations

class Rays:
    def __init__(self, origins, directions, maxDepth, currentDepth):
        self.origins = origins  # the point where the ray comes from
        self.directions = directions  # direction of the ray
        self.maxDepth = maxDepth  # ray_depth is the number of the refrections + transmissions/refractions, starting at zero for camera rays
        self.currentDepth = currentDepth

    def getColors(self, scene, light):

        print("depth",self.currentDepth)

        rayDirections = self.directions
        rayOrigins = self.origins

        maxRange = np.array((1e19,) * len(rayOrigins))
        interPoints = np.array(([0, 0, 0],) * len(rayOrigins))
        normalVectors = np.array(([0, 0, 0],) * len(rayOrigins))
        colors = np.array(([0, 0, 0],) * len(rayOrigins))
        bodies = np.zeros(len(rayOrigins))
        hitInformations = IntersecPointInformations(maxRange, interPoints, normalVectors, colors, bodies)

        time1 = time.time()
        for body in scene:
            #print(body)
            intersectionInformations = body.intersect(rayOrigins, rayDirections, True)
            hitInformations = getShortestDistancesInformations(hitInformations, intersectionInformations)
            #print(hitInformations)

        time2 = time.time() - time1
        print("intersecting took", time2)

        bodies = np.array(hitInformations.bodies)

        time1 = time.time()

        allIndices = []
        allBodyColors = []
        for body in scene:
            indices = np.where(bodies == body)[0]
            #("len ind", len(indices))

            zippedInformation = informationToZipped(hitInformations)
            #print("len all", len(zippedInformation))
            relevantInformations = zippedInformation[indices]
            #print("len rel",len(relevantInformations))
            if not len(relevantInformations):
                continue
            #print(len(indices))

            rDistances, rPoints, rNormalVectors, rColors, rBodies = zip(*relevantInformations)
            relevantInformation = IntersecPointInformations(rDistances, rPoints, rNormalVectors, rColors, rBodies)
            #print("len relevant:", len(relevantInformation.bodies))


            bodyColors = body.surface.getColor(indices, scene, relevantInformation, 3, self.currentDepth)
            #print(bodyColors)
            allIndices.extend(indices)
            allBodyColors.extend(bodyColors)


        allIndices = np.array(allIndices)
        allBodyColors = np.array(allBodyColors)
        sorter = np.argsort(allIndices)

        #print(allIndices)
        #print(sorter)
        colors = allBodyColors[sorter]
        #print(colors)
        #colors = getLambert(intersectionPoints, light, normalVectors, colors)
        time2 = time.time() - time1
        print("coloring took", time2)
        return colors

def getLambert(intersectionPoints, light, normalVectors, colors):
    raysToLightSource = np.subtract(light.origin, intersectionPoints)
    cos = np.array(
        [np.dot(ray, normalVector) / (np.linalg.norm(ray) * np.linalg.norm(normalVector)) for ray, normalVector in
         zip(raysToLightSource, normalVectors)])
    colors = np.array([np.multiply(color, a) for color, a in zip(colors, cos)])
    # print("1",colors)
    colors = colors * light.intensity * 0.18 / np.pi
    # print("2",colors)
    colors = np.array([color / (4 * np.pi * np.linalg.norm(ray)) for color, ray in zip(colors, raysToLightSource)])
    # print("3",colors)
    return colors



def informationToZipped(intersectionPointInformations):
    dist = intersectionPointInformations.distances
    insec = intersectionPointInformations.points
    normals = intersectionPointInformations.normalVectors
    disp = intersectionPointInformations.displacedPoints
    bodies = intersectionPointInformations.bodies

    return np.array(list(zip(dist, insec, normals, disp, bodies)))


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
