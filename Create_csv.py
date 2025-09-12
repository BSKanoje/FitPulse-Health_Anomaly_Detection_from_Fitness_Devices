import csv
import random
from datetime import datetime, timedelta


num_records = 100


start_date = datetime.now() - timedelta(days=num_records)
with open('fitness_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Time', 'SleepHours', 'HeartRate', 'Steps'])
    
    for i in range(num_records):
        date = (start_date + timedelta(days=i)).date()
        time = (datetime.min + timedelta(minutes=random.randint(0, 1439))).time().strftime('%H:%M:%S')
        sleep_hours = round(random.uniform(4, 10), 1)
        heart_rate = random.randint(50, 100)
        steps = random.randint(1000, 20000)
        writer.writerow([date, time, sleep_hours, heart_rate, steps])
