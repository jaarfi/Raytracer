import numpy as np


class Light:
    def __init__(self, origin, ambient, diffuse, specular, halfLength, intensity):
        self.origin = np.array(origin)
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.halfLength = halfLength
        self.intensity = intensity

        tempArray = np.array([[halfLength,0,halfLength],[-halfLength,0,halfLength],[halfLength,0,-halfLength],[-halfLength,0,-halfLength]])
        self.vertices = [origin+tempArray][0]
