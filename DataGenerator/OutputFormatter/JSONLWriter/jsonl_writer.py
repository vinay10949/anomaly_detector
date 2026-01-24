import json

class JSONLWriter:
    def __init__(self, filename):
        self.filename = filename

    def write(self, data):
        with open(self.filename, 'a') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')