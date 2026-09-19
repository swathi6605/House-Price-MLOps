import os
import pandas as pd


DATA_PATH = "data/processed/house_data.csv"

REQUIRED_COLUMNS = [
    "area_type",
    "availability",
    "location",
    "size",
    "society",
    "total_sqft",
    "bath",
    "balcony",
    "price"
]


def validate_data():

    print("Starting data validation...")

    # Check whether processed dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # 1. Check required columns
    print("\nChecking required columns...")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("All required columns are present.")

    # 2. Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # 3. Check duplicate rows
    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicate_count}")

    # 4. Check data types
    print("\nData types:")
    print(df.dtypes)

    # 5. Check target column
    if "price" not in df.columns:
        raise ValueError("Target column 'price' is missing.")

    print("\nTarget column 'price' is present.")

    # 6. Basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    print("\nData validation completed successfully.")

    return True


if __name__ == "__main__":
    validate_data()