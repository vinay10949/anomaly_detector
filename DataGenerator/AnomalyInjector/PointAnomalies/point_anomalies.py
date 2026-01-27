class PointAnomalies:
    def __init__(self, spike_factor=3.0):
        self.spike_factor = spike_factor

    def inject_spike(self, value, magnitude=None):
        # inject a spike, multiplied by magnitude (or default spike_factor)
        factor = magnitude if magnitude is not None else self.spike_factor
        return value * factor

    def inject_drop(self, value, drop_pct):
        # reduce value by percentage
        return value * (1 - drop_pct / 100.0)

    def inject_level_shift(self, data, start_idx, shift_amount):
        # add constant shift_amount to all points starting from start_idx
        for i in range(start_idx, len(data)):
            data[i]['value'] += shift_amount
        return data

    def inject(self, data):
        # inject point anomaly at random index
        import random
        if data:
            idx = random.randint(0, len(data)-1)
            data[idx]['value'] = self.inject_spike(data[idx]['value'])
        return data