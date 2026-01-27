import random

class NoiseGenerator:
    def __init__(self, distribution='gaussian', scale=1.0):
        self.distribution = distribution
        self.scale = scale

    def add_noise(self, value):
        if self.distribution == 'gaussian':
            noise = random.gauss(0, self.scale)
        elif self.distribution == 'uniform':
            noise = random.uniform(-self.scale, self.scale)
        elif self.distribution == 'laplace':
            # Laplace distribution: difference of two exponential random variables
            noise = self.scale * (random.expovariate(1) - random.expovariate(1))
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        return value + noise