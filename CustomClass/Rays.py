class Rays:
    def __init__(self, origins, directions, depth, n, reflections):
        self.origins = origins  # the point where the ray comes from
        self.directions = directions  # direction of the ray
        self.depth = depth  # ray_depth is the number of the refrections + transmissions/refractions, starting at zero for camera rays
        self.n = n  # ray_n is the index of refraction of the media in which the ray is travelling
        self.reflections = reflections