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
        rayDirections = np.array([ray / np.linalg.norm(ray) for ray in rayDirections])
        a = np.array([-(np.dot(ray, self.normalVector) + self.distanceToOrigin) for ray in rayOrigins])
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b == 0, -1, b)
        normalVectorArray = ((self.normalVector),) * len(rayOrigins)
        normalVectorArray = np.stack(normalVectorArray)

        colorsArray = (self.color, ) * len(rayOrigins)

        distances = a/b
        intersecPoints = [distance * direction + origin for distance, direction, origin in zip(distances,rayDirections,rayOrigins)]
        bodies = (self,) * len(rayOrigins)
        types = (type(self.surface),) * len(rayOrigins)
        #infos = IntersecPointInformations(a/b, intersecPoints, normalVectorArray, colorsArray, bodies, types)
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
        rayDirections = np.array([ray / np.linalg.norm(ray) for ray in rayDirections])
        a = np.array([-(np.dot(ray, self.normalVector) + self.distanceToOrigin) for ray in rayOrigins])
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b == 0, -1, b)
        normalVectorArray = (self.normalVector,) * len(rayOrigins)
        normalVectorArray = np.stack(normalVectorArray)
        #print("normalVectorArray:", normalVectorArray)
        colorsArray = (self.color, ) * len(rayOrigins)

        distances = a/b
        intersecPoints = [distance * direction + origin for distance, direction, origin in zip(distances,rayDirections,rayOrigins)]

        truthArray = [self.pointIsIn(point) for point in intersecPoints]
        distances = np.where(truthArray, distances, -1)
        bodies = (self,) * len(rayOrigins)
        types = (type(self.surface),) * len(rayOrigins)
        #infos = IntersecPointInformations(distances, intersecPoints, normalVectorArray, colorsArray, bodies,types)
        displacedPoints = intersecPoints + np.array(normalVectorArray)*0.01
        zippedInformation = np.array(list(zip(distances, intersecPoints, normalVectorArray, displacedPoints, colorsArray, bodies, types)))
        return zippedInformation

    def pointIsIn(self, intersectionPoint):
        if self.origin[0] - self.halfLength < intersectionPoint[0] < self.origin[0] + self.halfLength:
            if self.origin[2] - self.halfLength < intersectionPoint[2] < self.origin[2] + self.halfLength:
                return True
        return False
