import math

class SeasonalityModeler:
    def __init__(self):
        self.seasonal_period = 7 * 24 * 3600  # Weekly seasonality in seconds
        self.amplitude = 0.1  # 10% variation for visibility

    def model_seasonality(self, data):
        """
        Adds seasonal variations to the time series data.
        Uses a sinusoidal function based on the day of the week.
        """
        for d in data:
            # Calculate the phase based on the timestamp
            phase = (d['timestamp'] % self.seasonal_period) / self.seasonal_period * 2 * math.pi
            seasonal_factor = 1 + self.amplitude * math.sin(phase)
            d['value'] *= seasonal_factor
        return data