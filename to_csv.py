import json
import csv

def jsonl_to_csv_10k(jsonl_file, csv_file, limit=10000):
    with open(jsonl_file, 'r', encoding='utf-8') as infile, \
         open(csv_file, 'w', newline='', encoding='utf-8') as outfile:

        writer = csv.writer(outfile)
        writer.writerow(["timestamp", "value"])  # fixed header

        count = 0

        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)

            writer.writerow([
                data.get("timestamp"),
                data.get("value")
            ])

            count += 1
            if count >= limit:
                break

    print(f"Converted {count} rows.")

# Usage
jsonl_to_csv_10k("dataset.jsonl", "output.csv")
