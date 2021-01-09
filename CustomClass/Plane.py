import numpy as np
from .IntersecPointInformations import IntersecPointInformations


class Plane:
    def __init__(self, surface, normalVector, distanceToOrigin, shininess,
                 reflection):
        self.normalVector = normalVector / np.linalg.norm(normalVector)
        self.surface = surface
        self.distanceToOrigin = distanceToOrigin
        self.color = surface.color
        self.shininess = shininess
        self.reflection = reflection

    def intersect(self, rayOrigins, rayDirections, writeToDict):
        length = len(rayOrigins)

        rayDirectionsLengths = np.apply_along_axis(np.linalg.norm, 0, rayDirections)
        rayDirections = rayDirections / rayDirectionsLengths

        normalVector = self.normalVector

        a = rayOrigins.dot(normalVector)
        a = - a - self.distanceToOrigin
        b = np.dot(rayDirections, normalVector)
        b[b == 1] = -1
        normalVectorArray = ((normalVector),) * length

        colorsArray = (self.color, ) * length

        distances = a/b
        intersecPoints = np.add(rayDirections * distances[:,np.newaxis], rayOrigins)
        bodies = (self,) * length
        types = (type(self.surface),) * length
        displacedPoints = intersecPoints + np.array(normalVectorArray)*0.01
        zippedInformation = np.array(list(zip(distances, intersecPoints, normalVectorArray, displacedPoints, colorsArray, bodies, types)))
        return zippedInformation

class AxisAlignedSquare:
    def __init__(self, surface, origin, halfLength, shininess, reflection):
        self.surface = surface
        self.origin = origin
        self.halfLength = halfLength
        self.color = surface.color
        self.shininess = shininess
        self.reflection = reflection
        self.normalVector = [0,-1,0]
        self.distanceToOrigin = origin[1]
        self.vertices = [[self.origin[0] + halfLength, self.origin[1], self.origin[2] + halfLength],
                         [self.origin[0] - halfLength, self.origin[1], self.origin[2] + halfLength],
                         [self.origin[0] + halfLength, self.origin[1], self.origin[2] - halfLength],
                         [self.origin[0] - halfLength, self.origin[1], self.origin[2] - halfLength]]

    def intersect(self, rayOrigins, rayDirections, writeToDict):
        length = len(rayOrigins)

        rayDirectionsLengths = np.apply_along_axis(np.linalg.norm, 0, rayDirections)
        rayDirections = rayDirections / rayDirectionsLengths

        normalVector = self.normalVector

        a = rayOrigins.dot(normalVector)
        a = - a - self.distanceToOrigin
        b = np.dot(rayDirections, normalVector)
        b[b == 1] = -1
        normalVectorArray = ((normalVector),) * length

        colorsArray = (self.color,) * length

        distances = a / b
        intersecPoints = np.add(rayDirections * distances[:, np.newaxis], rayOrigins)

        #truthArray = [self.pointIsIn(point) for point in intersecPoints]
        distances = np.where(self.pointIsIn(intersecPoints), distances, -1)
        bodies = (self,) * len(rayOrigins)
        types = (type(self.surface),) * len(rayOrigins)
        #infos = IntersecPointInformations(distances, intersecPoints, normalVectorArray, colorsArray, bodies,types)
        displacedPoints = intersecPoints + np.array(normalVectorArray)*0.01
        zippedInformation = np.array(list(zip(distances, intersecPoints, normalVectorArray, displacedPoints, colorsArray, bodies, types)))
        return zippedInformation

    def pointIsIn(self, intersectionPoints):

        intersectionPointsX = intersectionPoints[:,0]
        intersectionPointsY = intersectionPoints[:,2]

        intersectionPointsXDistances = np.abs(intersectionPointsX - self.origin[0])
        intersectionPointsYDistances = np.abs(intersectionPointsY - self.origin[2])

        xTruthValues = np.where(intersectionPointsXDistances < self.halfLength, True, False)
        yTruthValues = np.where(intersectionPointsYDistances < self.halfLength, True, False)

        truthValues = np.logical_and(xTruthValues, yTruthValues)
        return truthValues
