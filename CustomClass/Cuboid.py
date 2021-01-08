import numpy as np
from .IntersecPointInformations import IntersecPointInformations


class Cuboid:
    def __init__(self, surface, size, center, rotation, shininess, reflection):
        self.size = size
        self.surface = surface
        self.rotation = np.radians(rotation)
        self.center = center

        self.color = surface.color
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

        colorsArray = (self.color, ) * len(rayOrigins)
        bodies = (self,) * len(rayOrigins)
        types = (type(self.surface),) * len(rayOrigins)
        infos = IntersecPointInformations(tReturn, intersecPointsWorldSpace, normalVectorArray, colorsArray, bodies, types)
        return infos