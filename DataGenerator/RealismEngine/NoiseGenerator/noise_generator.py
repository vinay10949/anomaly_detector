import random

class NoiseGenerator:
    def __init__(self):
        pass

    def add_noise(self, value):
        # add gaussian noise
        noise = random.gauss(0, 1)
        return value + noise