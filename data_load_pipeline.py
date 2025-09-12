import pandas as pd

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    return df

def data_quality_report(df):
    total_records = len(df)
    missing = df.isnull().sum().sum()
    quality_score = round(100 * (1 - missing / (df.shape[0] * df.shape[1])), 2) if df.shape[0] > 0 else 0
    report = {
        "total_records": total_records,
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe(include='all').to_dict(),
        "quality_score": f"{quality_score}%"
    }
    return report

def quality_check(df):
    errors = []
    # Handle missing columns gracefully
    for col in ['SleepHours', 'HeartRate', 'Steps']:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            df[col] = pd.NA

    sleep_min, sleep_max = 4, 10
    hr_min, hr_max = 50, 100
    steps_min, steps_max = 1000, 20000

    sleep_err = ~df['SleepHours'].between(sleep_min, sleep_max)
    hr_err = ~df['HeartRate'].between(hr_min, hr_max)
    steps_err = ~df['Steps'].between(steps_min, steps_max)

    if sleep_err.any():
        errors.append("SleepHours out of range (4-10 hours).")
    if hr_err.any():
        errors.append("HeartRate out of range (50-100 bpm).")
    if steps_err.any():
        errors.append("Steps out of range (1000-20000).")

    error_rows = df[sleep_err | hr_err | steps_err]
    return errors, error_rows

def run_pipeline(file_path):
    df = load_data(file_path)
    report = data_quality_report(df)
    errors, error_rows = quality_check(df)

    print(f"\nData Quality Report for {file_path}:")
    print(f"Total Records: {report['total_records']}")
    print(f"Shape: {report['shape']}")
    print(f"Missing Values: {report['missing']}")
    print(f"Quality Score: {report['quality_score']}")
    # Optionally print describe: print(f"Describe: {report['describe']}")

    if not errors:
        print(f"\n{file_path} - No errors found. Showing 5 data lines:\n")
        print(df.head())
    else:
        print(f"\n{file_path} - Data Quality Errors Detected:")
        for err in errors:
            print("-", err)
        print("\nRows with errors:\n", error_rows)

if __name__ == "__main__":
    for file in ['fitness_data.csv', 'fitness_data.json']:
        run_pipeline(file)
