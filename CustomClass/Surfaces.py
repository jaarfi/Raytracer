import numpy as np
from .Rays import Rays

class Matt:
    def __init__(self, color, nrReflectedRays = 20):
        self.color = color
        self.nrReflectedRays = nrReflectedRays


    def getColor(self, rayOrigins, scene, hitInformation, maxDepth, currentDepth):

        colors = ([0,0,0],) * len(rayOrigins)
        colorsArray = (self.color,) * len(rayOrigins)

        normalVectorsExtended = np.tile(hitInformation.normalVectors, (self.nrReflectedRays,1))
        #normalVectorsExtended = hitInformation.normalVectors
        randomVectorArray = randomVectors(normalVectorsExtended)
        #print("random",randomVectorArray)
        displacedPointsExtended = np.tile(hitInformation.displacedPoints, (self.nrReflectedRays,1))
        #displacedPointsExtended = hitInformation.displacedPoints

        rays = Rays(displacedPointsExtended, randomVectorArray, maxDepth, currentDepth+1)
        #print("len colors:", len(colors), "len rays:",len(rays.origins))

        if currentDepth < 1:
            temp = rays.getColors(scene,0)
            #print("temp:",temp)
            #print("slice:",temp[0::len(rayOrigins)])

            # = []
            #for i in range(rayOrigins):
            #    t = np.array(temp[i::len(rayOrigins)])
            #    shadows.append(t.mean())

            temp = np.array([np.array(np.array(temp)[i::len(rayOrigins)]).mean(axis=0) for i in range(len(rayOrigins))])
            colors += temp

        return colors



class Radiant:
    def __init__(self, color):
        self.color = color

    def getColor(self, rayOrigins, scene, hitInformation, maxDepth, currentDepth):
        print("hallo")
        colors = (self.color,) * len(rayOrigins)
        return colors


def randomVectors(normalVectors):
    randomVectors = np.random.rand(len(normalVectors),3)
    randomVectors = np.array([vec / np.linalg.norm(vec) for vec in randomVectors])
    randomVectors = np.array([vec * (np.random.rand(1) * 2 - 1) for vec in randomVectors])

    randomVectors = randomVectors + normalVectors
    #print("randomVectors:", randomVectors)
    return randomVectors
