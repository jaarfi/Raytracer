import numpy as np
import CustomClass.Rays

class Matt:
    def __init__(self, color, nrReflectedRays =  20):
        self.color = color
        self.nrReflectedRays = nrReflectedRays


    def getColor(self, rayOrigins, scene, hitInformation, maxDepth, currentDepth):

        colors = ([0,0,0],) * len(rayOrigins)
        #colorsArray = np.array((self.color,) * len(rayOrigins))
        colorsArray = hitInformation.colors




        if currentDepth < 1:
            normalVectors = hitInformation.normalVectors
            #print("normals", normalVectors)
            #print(normalVectors.shape)
            normalVectorsExtended = np.tile(normalVectors, (self.nrReflectedRays, 1))
            #normalVectorsExtended = np.repeat(normalVectors, self.nrReflectedRays)
            randomVectorArray = hemisphereVectors(normalVectorsExtended)


            displacedPointsExtended = np.tile(hitInformation.displacedPoints, (self.nrReflectedRays, 1))
            rays = CustomClass.Rays.Rays(displacedPointsExtended, randomVectorArray, maxDepth, currentDepth + 1)

            dotProds = np.einsum('ij,ij->i',randomVectorArray, normalVectorsExtended)
            dotProds = np.array([np.array(np.array(dotProds)[i::len(rayOrigins)]).mean(axis=0) for i in range(len(rayOrigins))])

            temp = rays.getColors(scene)
            temp = np.array([np.array(np.array(temp)[i::len(rayOrigins)]).mean(axis=0) for i in range(len(rayOrigins))])
            temp = temp * dotProds[:,None] / np.pi * 0.18
            colors += np.multiply(colorsArray,temp)

        elif currentDepth < maxDepth:
            normalVectors = hitInformation.normalVectors
            randomVectorArray = randomVectors(normalVectors)
            rays = CustomClass.Rays.Rays(hitInformation.displacedPoints, randomVectorArray, maxDepth, currentDepth + 1)

            dotProds = np.einsum('ij,ij->i', randomVectorArray, normalVectors)
            temp = rays.getColors(scene)
            temp = temp * dotProds[:, None] / np.pi
            colors += np.multiply(colorsArray, temp)

        return colors



class Radiant:
    def __init__(self, color):
        self.color = color

    def getColor(self, rayOrigins, scene, hitInformation, maxDepth, currentDepth):
        colors = (self.color,) * len(rayOrigins)
        colorsArray = hitInformation.colors
        return colorsArray


def randomVectors(normalVectors):
    randomVectors = np.random.rand(len(normalVectors),3)
    randomVectors = np.array([vec / np.linalg.norm(vec) for vec in randomVectors])
    randomVectors = np.array([vec * (np.random.rand(1) * 2 - 1) for vec in randomVectors])

    randomVectors = randomVectors + normalVectors
    #print("randomVectors:", randomVectors)
    return randomVectors

def hemisphereVectors(normalVectors):
    zeros = np.zeros(len(normalVectors))
    option1 = np.array([normalVectors[:,2],zeros,-normalVectors[:,0]]).T
    #option1 = np.array([[vec[2],0, --vec[0]] for vec in normalVectors])
    option2 = np.array([zeros,-normalVectors[:,2],normalVectors[:,1]]).T
    #option2 = np.array([[0,-vec[2],vec[1]] for vec in normalVectors])
    Nts = np.where((np.abs(normalVectors[:,0]) > np.abs(normalVectors[:,1]))[...,None], option1, option2)
    #print("1:", option1, "\n2:", option2, "\np:", perpendicularVectors)
    Nbs = np.cross(normalVectors,Nts)

    rand = np.random.rand(len(Nts), 2)

    sinThetas = np.sqrt(np.ones(len(Nts)) - np.power(rand[:,0],2))
    phis = 2 * np.pi * rand[:,1]

    xs = np.multiply(sinThetas, np.cos(phis))
    ys = rand[:,0]
    zs = np.multiply(sinThetas, np.sin(phis))

    xsWorld = np.multiply(xs, Nbs[:,0]) +  np.multiply(ys, normalVectors[:,0]) +  np.multiply(zs, Nts[:,0])
    ysWorld = np.multiply(xs, Nbs[:,1]) +  np.multiply(ys, normalVectors[:,1]) +  np.multiply(zs, Nts[:,1])
    zsWorld = np.multiply(xs, Nbs[:,2]) +  np.multiply(ys, normalVectors[:,2]) +  np.multiply(zs, Nts[:,2])

    vectors = np.transpose(np.append(np.array([xsWorld, ysWorld]), [zsWorld], axis = 0))

    return vectors