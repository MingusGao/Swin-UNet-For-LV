import pandas as pd
import os

original_csv_path = '../data/echonet_peds/VolumeTracings.csv'  

cleaned_csv_path = original_csv_path.replace('.csv', '_cleaned.csv')

numeric_columns_to_clean = [
    'X',
    'Y',
    'Frame'
]

print("--- runing clean script ---")

if not os.path.exists(original_csv_path):
    print(f"warn:cannot find csv！")
    print(f"please check path: {original_csv_path}")
else:
    print(f"Loading csv: {original_csv_path}")

    df = pd.read_csv(original_csv_path)
    original_rows = len(df)
    print(f" load successful，raw file include {original_rows} rows。")

    print(f"\n checking data: {numeric_columns_to_clean}")


    for col in numeric_columns_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f" Warning：cloum '{col}' not exsit，skip")

    df_cleaned = df.dropna(subset=numeric_columns_to_clean)
    cleaned_rows = len(df_cleaned)

    print("\n--- clean result ---")
    print(f"raw rows: {original_rows}")
    print(f"cleaned rows: {cleaned_rows}")
    print(f"original_rows - cleaned_rows: {original_rows - cleaned_rows}")
    print("--------------------")

    df_cleaned.to_csv(cleaned_csv_path, index=False)

    print(f"\n Successful! Saved:")
    print(f"{cleaned_csv_path}")