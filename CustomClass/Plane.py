import numpy as np
from .IntersecPointInformations import IntersecPointInformations


class Plane:
    def __init__(self, normalVector, distanceToOrigin, colorAmbient, colorDiffuse, colorSpecular, shininess,
                 reflection):
        self.normalVector = normalVector / np.linalg.norm(normalVector)
        self.distanceToOrigin = distanceToOrigin
        self.colorAmbient = colorAmbient
        self.colorDiffuse = colorDiffuse
        self.colorSpecular = colorSpecular
        self.shininess = shininess
        self.reflection = reflection

    def intersect(self, rayOrigins, rayDirections, writeToDict):
        rayDirections = np.array([ray / np.linalg.norm(ray) for ray in rayDirections])
        a = np.array([-(np.dot(ray, self.normalVector) + self.distanceToOrigin) for ray in rayOrigins])
        b = np.dot(rayDirections, self.normalVector)

        b = np.where(b == 0, -1, b)
        normalVectorArray = (self.normalVector,) * len(rayOrigins)
        #print("normalVectorArray:", normalVectorArray)
        colorsArray = (self.colorDiffuse, ) * len(rayOrigins)

        distances = a/b
        intersecPoints = [distance * direction + origin for distance, direction, origin in zip(distances,rayDirections,rayOrigins)]
        print("distances:", distances)
        print("insecpoints:", intersecPoints)
        infos = IntersecPointInformations(a/b, intersecPoints, normalVectorArray, colorsArray)
        return infos