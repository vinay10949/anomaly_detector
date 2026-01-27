class TimingAnomalies:
    def __init__(self):
        pass

    def inject_lag(self, data_segment, lag_steps):
        # Shifts the values in the segment forward by lag_steps
        # Values are moved to later positions, earlier positions hold their original or set to 0
        import random
        if not data_segment or lag_steps <= 0:
            return data_segment
        # To avoid overwriting, create a copy of values
        original_values = [d['value'] for d in data_segment]
        for i in range(len(data_segment)):
            if i < lag_steps:
                # For the first lag_steps, perhaps set to 0 or keep original
                data_segment[i]['value'] = 0  # simulate lag by setting early values to 0
            else:
                data_segment[i]['value'] = original_values[i - lag_steps]
        return data_segment

    def inject_missed_beat(self, data, idx):
        # Sets a value to 0 or holds the previous value when a periodic peak was expected
        import random
        if 0 <= idx < len(data):
            # Randomly choose to set to 0 or hold previous
            if random.random() < 0.5 and idx > 0:
                data[idx]['value'] = data[idx - 1]['value']  # hold previous
            else:
                data[idx]['value'] = 0  # missed beat
        return data

    def inject(self, data):
        # inject timing anomaly: e.g., shift a timestamp
        import random
        if data:
            idx = random.randint(0, len(data)-1)
            data[idx]['timestamp'] += random.randint(60, 300)  # delay by 1-5 minutes
        return data