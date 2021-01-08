import numpy as np
import time
from .IntersecPointInformations import IntersecPointInformations

class Rays:
    def __init__(self, origins, directions, maxDepth, currentDepth):
        self.origins = origins  # the point where the ray comes from
        self.directions = directions  # direction of the ray
        self.maxDepth = maxDepth  # ray_depth is the number of the refrections + transmissions/refractions, starting at zero for camera rays
        self.currentDepth = currentDepth

    def getColors(self, scene):

        print("depth",self.currentDepth)

        rayDirections = self.directions
        rayOrigins = self.origins

        zippedHitInformations = np.array([informationToZipped(body.intersect(rayOrigins, rayDirections, True)) for body in scene])
        zippedHitInformations = reduceToMinimum(zippedHitInformations)
        bodies = zippedHitInformations[:,len(zippedHitInformations[0])-2]


        allIndices = []
        allBodyColors = []
        for body in scene:
            indices = np.where(bodies == body)[0]
            relevantInformations = zippedHitInformations[indices]
            if not len(relevantInformations):
                continue

            rDistances, rPoints, rNormalVectors, rColors, rBodies, rTypes = zip(*relevantInformations)
            relevantInformation = IntersecPointInformations(rDistances, rPoints, rNormalVectors, rColors, rBodies, rTypes)


            bodyColors = body.surface.getColor(indices, scene, relevantInformation, self.maxDepth, self.currentDepth)
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
    colors = intersectionPointInformations.colors
    bodies = intersectionPointInformations.bodies
    types = intersectionPointInformations.types

    return np.array(list(zip(dist, insec, normals, colors, bodies, types)))


def reduceToMinimum(hitIntersectionInformations):
    distances = hitIntersectionInformations[:,:,0]
    distances = np.transpose(distances)
    distances[distances < 0] = np.inf

    indices = np.argmin(distances,axis=1)
    countIndices = np.arange(len(indices))

    finalZip = hitIntersectionInformations[indices,countIndices]
    return finalZip

