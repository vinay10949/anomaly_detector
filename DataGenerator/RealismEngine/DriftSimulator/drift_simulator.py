import random

class DriftSimulator:
    def __init__(self, drift_rate=0.0001, noise_level=0.01):
        self.drift_rate = drift_rate  # Rate of drift per second
        self.noise_level = noise_level  # Additional noise for drift

    def simulate_drift(self, data):
        """
        Simulates concept drift by adding a linear trend over time.
        The drift increases the value gradually as time progresses.
        """
        if not data:
            return data

        # Find the time range
        timestamps = [d['timestamp'] for d in data]
        start_time = min(timestamps)
        end_time = max(timestamps)
        time_range = end_time - start_time

        for d in data:
            # Calculate progress (0 to 1)
            progress = (d['timestamp'] - start_time) / time_range if time_range > 0 else 0
            # Linear drift
            drift = self.drift_rate * (d['timestamp'] - start_time)
            # Add some noise to the drift
            noise = random.gauss(0, self.noise_level)
            total_drift = drift + noise
            d['value'] += total_drift

        return data