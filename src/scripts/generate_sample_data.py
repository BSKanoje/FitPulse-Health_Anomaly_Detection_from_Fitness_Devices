import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
start_time = datetime(2023, 1, 1)
timestamps = [start_time + timedelta(minutes=i) for i in range(10080)]  # 1 week minutes

heart_rate = np.clip(60 + 10 * np.sin(np.arange(10080) / 1440 * 2 * np.pi) + np.random.normal(0, 5, 10080), 50, 120)
steps = np.random.poisson(50, 10080)
sleep_duration = np.where((np.arange(10080) % 1440 >= 1320) | (np.arange(10080) % 1440 < 60), np.random.uniform(5, 10, 10080), 0)

# Inject anomalies for testing
heart_rate[5000] = 180  # Point spike
heart_rate[2000:2010] = 110  # Contextual during sleep
sleep_duration[1000:1060] = 2  # Collective low sleep
steps[3000:3100] = 0  # Collective inactivity

df = pd.DataFrame({'timestamp': timestamps, 'heart_rate': heart_rate, 'steps': steps, 'sleep_duration': sleep_duration})
df.to_csv('sample_data.csv', index=False)
print("Sample data generated: sample_data.csv (with injected anomalies)")