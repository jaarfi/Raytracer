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

        length = len(rayOrigins)

        rayDirections4 = np.insert(np.array(rayDirections), 3, 0, axis=1)
        rayDirectionsBoxSpace = np.einsum('...ij,...j', self.transformationMatrix, rayDirections4)
        rayDirectionsBoxSpace = np.delete(rayDirectionsBoxSpace, 3, 1)

        rayDirectionsBoxSpaceLengths = np.apply_along_axis(np.linalg.norm, 0, rayDirectionsBoxSpace)

        rayDirectionsBoxSpace = rayDirectionsBoxSpace / rayDirectionsBoxSpaceLengths

        rayOrigins4 = np.insert(np.array(rayOrigins), 3, 1, axis=1)
        rayOriginsBoxSpace = np.einsum('...ij,...j', self.transformationMatrix, rayOrigins4)
        rayOriginsBoxSpace = np.delete(rayOriginsBoxSpace, 3, 1)

        #rayDirectionsBoxSpaceNoZeros = rayDirectionsBoxSpace[rayDirectionsBoxSpace == 0] = 0.00001
        rayDirectionsBoxSpaceNoZeros = rayDirectionsBoxSpace
        rayDirectionsBoxSpaceNoZeros[rayDirectionsBoxSpaceNoZeros == 0] = 0.00001
        #rayDirectionsBoxSpaceNoZeros = np.array([np.where(a == 0, 0.000001, a) for a in rayDirectionsBoxSpace])

        rayOriginsBoxSpacePlus = - rayOriginsBoxSpace + self.size
        t1 = np.divide(rayOriginsBoxSpacePlus, rayDirectionsBoxSpaceNoZeros)

        rayOriginsBoxSpaceMinus = - rayOriginsBoxSpace - self.size
        t2 = np.divide(rayOriginsBoxSpaceMinus, rayDirectionsBoxSpaceNoZeros)
        t3 = np.minimum(t1,t2)
        t4 = np.maximum(t1,t2)

        tMin = np.max(t3, axis=1)
        tMax = np.min(t4, axis=1)

        tReturn = np.where(np.logical_and(tMin < tMax, tMax > 0), tMin, -1)

        tReturn = np.where(tMin < 0, tMax, tReturn)

        t = np.block([[t1[:,0]], [t2[:,0] ],[t1[:,1]], [t2[:,1]],[t1[:,2]], [t2[:,2]]]).T

        ones = np.ones(length)
        ones *= -1
        t = np.insert(t,6,ones,axis=1)


        #tSides = [np.where(a == b)[0] for a, b in zip(t, tReturn)]
        #tSides = [a[0] for a in tSides]
        tSides = np.where(t == np.array(tReturn)[:, None])[1]

        normalVectors = np.array((self.transformationMatrix[0][:3],  # rechts
                                  - 1 * self.transformationMatrix[0][:3],  # links
                                  self.transformationMatrix[1][:3],  # oben
                                  -1 * self.transformationMatrix[1][:3],  # unten
                                  self.transformationMatrix[2][:3],  # //hinten
                                  -1 * self.transformationMatrix[2][:3],
                                 [0,0,0]))  # vorne


        intersecPointsWorldSpace = rayOrigins + tReturn[..., None] * rayDirections

        normalVectorArray = [normalVectors[tSides]][0]
        #normalVectorArray = np.stack(normalVectorArray)

        colorsArray = (self.color, ) * length
        bodies = (self,) * length
        types = (type(self.surface),) * length
        displacedPoints = intersecPointsWorldSpace + np.array(normalVectorArray)*0.01
        zippedInformation = np.array(list(zip(tReturn, intersecPointsWorldSpace, normalVectorArray, displacedPoints, colorsArray, bodies, types)))
        #infos = IntersecPointInformations(tReturn, intersecPointsWorldSpace, normalVectorArray, colorsArray, bodies, types)
        return zippedInformation