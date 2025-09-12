import pandas as pd

# File name of the JSON file generated earlier
filename = "fitness_data.json"

try:
    # Load JSON into a pandas DataFrame
    df = pd.read_json(filename)

    print(f"JSON file '{filename}' loaded successfully!")
    print("First 5 records:")
    print(df.head())

    # Show basic info
    print("\nData Summary:")
    print(df.describe())

except FileNotFoundError:
    print(f"Error: File '{filename}' not found. Please run 'create_json.py' first.")
