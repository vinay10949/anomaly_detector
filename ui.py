import json
import csv

def load_data(filename):
    data = []
    if filename.endswith('.jsonl'):
        with open(filename, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif filename.endswith('.csv'):
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                row['timestamp'] = int(row['timestamp'])
                row['value'] = float(row['value'])
                data.append(row)
    else:
        raise ValueError("Unsupported file format. Use .jsonl or .csv")
    return data

def load_metadata(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def display_time_series(data, anomalies):
    # Simple text display
    anomaly_dict = { (a['timestamp'], a['entity_id'], a['metric']): a for a in anomalies }
    anomaly_count = 0
    anomaly_types = {}
    for d in data:
        key = (d['timestamp'], d['entity_id'], d['metric'])
        if key in anomaly_dict:
            anomaly_type = anomaly_dict[key].get('anomaly_type', 'unknown')
            print(f"ANOMALY ({anomaly_type}): {d}")
            anomaly_count += 1
            anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
    print(f"Total Anomalies: {anomaly_count}")
    print("Anomaly Types Breakdown:")
    for typ, cnt in anomaly_types.items():
        print(f"  {typ}: {cnt}")

if __name__ == "__main__":
    # Try to load dataset.jsonl or dataset.csv
    try:
        data = load_data('dataset.jsonl')
        dataset_file = 'dataset.jsonl'
    except FileNotFoundError:
        try:
            data = load_data('dataset.csv')
            dataset_file = 'dataset.csv'
        except FileNotFoundError:
            print("No dataset file found. Run main.py first to generate data.")
            exit(1)
    
    try:
        anomalies = load_metadata('metadata.json')
    except FileNotFoundError:
        print("No metadata file found. Anomalies will not be highlighted.")
        anomalies = []
    
    print(f"Time Series Data with Anomalies from {dataset_file}:")
    display_time_series(data, anomalies)