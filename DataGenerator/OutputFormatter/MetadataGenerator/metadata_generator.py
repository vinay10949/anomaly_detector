import json

class MetadataGenerator:
    def __init__(self):
        self.anomalies = []

    def add_anomaly(self, timestamp, entity_id, signal_type, metric, entity_type=None, anomaly_type=None):
        anomaly = {
            'timestamp': timestamp,
            'entity_id': entity_id,
            'signal_type': signal_type,
            'metric': metric
        }
        if entity_type:
            anomaly['entity_type'] = entity_type
        if anomaly_type:
            anomaly['anomaly_type'] = anomaly_type
        self.anomalies.append(anomaly)

    def get_metadata(self):
        return self.anomalies

    def save_metadata(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.anomalies, f)