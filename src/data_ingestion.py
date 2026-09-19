import os
import pandas as pd


RAW_DATA_PATH = "data/raw/Bengaluru_House_Data.csv"
PROCESSED_DATA_PATH = "data/processed/house_data.csv"


def ingest_data():
    print("Starting data ingestion...")

    # Check whether raw dataset exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    # Read raw dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Save a copy for the next pipeline stage
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    ingest_data()