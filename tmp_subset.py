import ast
from pathlib import Path

input_file = Path("data/cat_62_sweden_full_week1.txt")
mini_file = Path("data/cat_62_sweden_mini.txt")

first_time = None
records = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            r = ast.literal_eval(line)
            t = r['time']
            if first_time is None:
                first_time = t
            if t >= first_time and t <= first_time + 600:
                records.append(r)
            if len(records) > 5000: # safety cap
                break

records.sort(key=lambda x: x['time'])
with open(mini_file, 'w') as f:
    for r in records:
        f.write(str(r) + "\n")

if records:
    print(f"Collected {len(records)} records.")
    print(f"T0: {records[0]['time']}, T1: {records[-1]['time']}, Diff: {records[-1]['time']-records[0]['time']} s")
else:
    print("No records found.")
