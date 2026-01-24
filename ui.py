import json

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_metadata(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def display_time_series(data, anomalies):
    # Simple text display
    anomaly_dict = { (a['timestamp'], a['entity_id'], a['metric']): a for a in anomalies }
    anomaly_count = 0
    for d in data:
        key = (d['timestamp'], d['entity_id'], d['metric'])
        if key in anomaly_dict:
            print(f"ANOMALY: {d}")
            anomaly_count += 1
    print(f"Total Anomalies: {anomaly_count}")

if __name__ == "__main__":
    data = load_data('dataset.jsonl')
    anomalies = load_metadata('metadata.json')
    print("Time Series Data with Anomalies:")
    display_time_series(data, anomalies)