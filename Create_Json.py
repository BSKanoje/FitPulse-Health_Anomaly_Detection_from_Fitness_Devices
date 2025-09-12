import json
import random
from datetime import datetime, timedelta

num_records = 100

data = []
start_date = datetime.now() - timedelta(days=num_records)
for i in range(num_records):
    date = (start_date + timedelta(days=i)).date().isoformat()
    time = (datetime.min + timedelta(minutes=random.randint(0, 1439))).time().strftime('%H:%M:%S')
    record = {
        "Date": date,
        "Time": time,
        "SleepHours": round(random.uniform(4, 10), 1),
        "HeartRate": random.randint(50, 100),
        "Steps": random.randint(1000, 20000)
    }
    data.append(record)

with open('fitness_data.json', 'w') as file:
    json.dump(data, file, indent=4)
