import csv
import random
from datetime import datetime, timedelta

# Number of records to generate
NUM_RECORDS = 100

# Start time (current time - NUM_RECORDS minutes)
start_time = datetime.now() - timedelta(minutes=NUM_RECORDS)

# Output CSV file
filename = "data/fitness_data.csv"

# Generate random fitness data
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "heart_rate", "sleep_duration", "step_count"])  # Header

    for i in range(NUM_RECORDS):
        timestamp = (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        heart_rate = random.randint(55, 120)           # beats per minute
        sleep_duration = random.uniform(0, 10)         # hours (per session)
        step_count = random.randint(0, 200)            # steps per minute

        writer.writerow([timestamp, heart_rate, round(sleep_duration, 2), step_count])

print(f"CSV file '{filename}' created with {NUM_RECORDS} records.")
