import pandas as pd

# File name of the CSV file generated earlier
filename = "fitness_data.csv"

try:
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(filename)

    print(f"CSV file '{filename}' loaded successfully!")
    print("First 5 records:")
    print(df.head())

    # Show basic info
    print("\nData Summary:")
    print(df.describe())

except FileNotFoundError:
    print(f"Error: File '{filename}' not found. Please run 'create_csv.py' first.")
