class ContextualAnomalies:
    def __init__(self):
        pass

    def inject_contextual(self, data_segment, global_mean):
        # Creates an anomaly where a point is far from the local moving average but within global bounds
        import random
        if not data_segment:
            return data_segment
        # Compute moving average with window size 5
        window_size = 5
        moving_avgs = []
        for i in range(len(data_segment)):
            start = max(0, i - window_size // 2)
            end = min(len(data_segment), i + window_size // 2 + 1)
            avg = sum(d['value'] for d in data_segment[start:end]) / (end - start)
            moving_avgs.append(avg)
        
        # Pick a random index
        idx = random.randint(0, len(data_segment)-1)
        local_avg = moving_avgs[idx]
        
        # Create anomaly: set value far from local_avg but within global bounds
        # Assuming global bounds are roughly global_mean ± (global_mean - local_avg) or something
        deviation = abs(global_mean - local_avg)
        if local_avg < global_mean:
            # Local is low, set high but not beyond global_mean + deviation
            new_value = global_mean + deviation * random.uniform(0.5, 1.5)
        else:
            # Local is high, set low but not below global_mean - deviation
            new_value = global_mean - deviation * random.uniform(0.5, 1.5)
        
        data_segment[idx]['value'] = new_value
        return data_segment

    def inject(self, data):
        # inject contextual anomaly: e.g., a value that is anomalous given the context (simple: set to half)
        import random
        if data:
            idx = random.randint(0, len(data)-1)
            data[idx]['value'] *= 0.5  # dip, assuming context expects higher
        return data