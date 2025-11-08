import pandas as pd

def preprocess_data(df):
    """
    Preprocess fitness data:
    - Ensure timestamp is datetime
    - Sort by timestamp
    - Fill missing values if needed
    """
    # Handle missing timestamp column
    if 'timestamp' not in df.columns:
        df.insert(0, 'timestamp', pd.date_range(start='2025-01-01', periods=len(df), freq='H'))

    # Convert to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])

    # Sort
    df = df.sort_values('timestamp')

    # Optional: fill missing metrics with forward fill
    for col in ['heart_rate', 'steps', 'sleep_duration']:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')

    df = df.set_index('timestamp')

    df = df.reset_index()

    return df
