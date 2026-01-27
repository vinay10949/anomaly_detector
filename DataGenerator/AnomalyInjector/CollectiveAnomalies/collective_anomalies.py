class CollectiveAnomalies:
    def __init__(self):
        pass

    def inject_variance_explosion(self, data_segment, multiplier):
        # drastically increases the variance/noise for a sequence of points
        import random
        import statistics
        if not data_segment:
            return data_segment
        values = [d['value'] for d in data_segment]
        mean_val = statistics.mean(values)
        # Assume base noise std is 1, increase by multiplier
        base_std = 1.0
        for d in data_segment:
            noise = random.gauss(0, base_std * multiplier)
            d['value'] += noise
        return data_segment

    def inject_trend_reversal(self, data_segment):
        # If the segment has a trend, reverse it
        import statistics
        if len(data_segment) < 2:
            return data_segment
        n = len(data_segment)
        x = list(range(n))
        y = [d['value'] for d in data_segment]
        # Calculate slope (simple linear regression)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi**2 for xi in x)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        
        if abs(slope) > 0.01:  # if there's a trend
            # Reverse the trend: adjust each point
            for i, d in enumerate(data_segment):
                predicted = intercept + slope * i
                # Reverse: instead of predicted, set to intercept + (-slope) * i, but to reverse relative to mean
                mean_y = statistics.mean(y)
                d['value'] = mean_y + (mean_y - predicted)
        return data_segment

    def inject_square_wave_distortion(self, data_segment):
        # Replace a smooth curve with a square-wave approximation
        import statistics
        if not data_segment:
            return data_segment
        values = [d['value'] for d in data_segment]
        mean_val = statistics.mean(values)
        min_val = min(values)
        max_val = max(values)
        # Approximate with square wave: alternate high and low
        period = max(2, len(data_segment) // 4)  # e.g., 4 periods
        for i, d in enumerate(data_segment):
            if (i // period) % 2 == 0:
                d['value'] = max_val
            else:
                d['value'] = min_val
        return data_segment

    def inject(self, data):
        # inject a collective anomaly: a cluster of spikes
        import random
        if len(data) > 10:
            start_idx = random.randint(0, len(data)-10)
            end_idx = start_idx + random.randint(3, 10)
            for i in range(start_idx, min(end_idx, len(data))):
                data[i]['value'] *= random.uniform(2, 4)  # spike
        return data