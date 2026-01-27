import json
import csv

class JSONLWriter:
    def __init__(self, filename):
        self.filename = filename

    def write(self, data):
        with open(self.filename, 'a') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

class CSVWriter:
    def __init__(self, filename):
        self.filename = filename

    def write(self, data):
        if not data:
            return
        fieldnames = data[0].keys()
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Write header only if file is empty
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(data)