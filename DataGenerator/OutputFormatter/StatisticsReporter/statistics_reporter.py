import statistics

class StatisticsReporter:
    def __init__(self):
        pass

    def report(self, data):
        values = [d['value'] for d in data]
        print(f"Mean: {statistics.mean(values)}")
        print(f"Std: {statistics.stdev(values)}")
        print(f"Min: {min(values)}")
        print(f"Max: {max(values)}")
        print(f"Count: {len(values)}")