import json
import random
from datetime import datetime, timedelta

# Number of records to generate
NUM_RECORDS = 100

# Start time (current time - NUM_RECORDS minutes)
start_time = datetime.now() - timedelta(minutes=NUM_RECORDS)

# Output JSON file
filename = "fitness_data.json"

data = []

# Generate random fitness data
for i in range(NUM_RECORDS):
    timestamp = (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
    heart_rate = random.randint(55, 120)           # beats per minute
    sleep_duration = random.uniform(0, 10)         # hours (per session)
    step_count = random.randint(0, 200)            # steps per minute

    data.append({
        "timestamp": timestamp,
        "heart_rate": heart_rate,
        "sleep_duration": round(sleep_duration, 2),
        "step_count": step_count
    })

# Save to JSON file
with open(filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"✅ JSON file '{filename}' created with {NUM_RECORDS} records.")
