import random

class MetricGenerators:
    def __init__(self, entity_profiles):
        self.profiles = entity_profiles

    def generate_value(self, entity_type, signal_type, metric):
        profile = self.profiles.get_profile(entity_type, signal_type, metric)
        if profile:
            mean = profile['mean']
            std = profile['std']
            return random.gauss(mean, std)
        else:
            return random.uniform(0, 100)  # default