import os
import re
import pandas as pd


INPUT_PATH = "data/processed/house_data.csv"
OUTPUT_PATH = "data/processed/cleaned_house_data.csv"


def convert_sqft_to_num(value):
    """
    Convert total_sqft values into numeric square feet.

    Examples:
        '1200' -> 1200.0
        '1000 - 1500' -> 1250.0
        '34.46Sq. Meter' -> 371.0 approximately
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Handle ranges such as "1000 - 1500"
    if "-" in value:
        parts = value.split("-")

        try:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        except ValueError:
            return None

    # Handle square meter values
    if "Sq. Meter" in value:
        try:
            number = float(
                re.findall(r"\d+\.?\d*", value)[0]
            )
            return number * 10.7639
        except (ValueError, IndexError):
            return None

    # Handle square yards
    if "Sq. Yards" in value:
        try:
            number = float(
                re.findall(r"\d+\.?\d*", value)[0]
            )
            return number * 9
        except (ValueError, IndexError):
            return None

    # Normal numeric value
    try:
        return float(value)
    except ValueError:
        return None


def extract_bhk(value):
    """
    Extract BHK/Bedroom number from size.
    Example: '3 BHK' -> 3
             '4 Bedroom' -> 4
             '1 RK' -> 1
    """

    if pd.isna(value):
        return None

    match = re.search(r"\d+", str(value))

    if match:
        return int(match.group())

    return None


def clean_data():

    print("Starting data cleaning...")

    # Check input file
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    # Load data
    df = pd.read_csv(INPUT_PATH)

    print(f"Original rows: {len(df)}")

    # Remove duplicate rows
    df = df.drop_duplicates()

    print(f"After removing duplicates: {len(df)}")

    # Remove rows where location is missing
    df = df.dropna(subset=["location"])

    # Create numeric BHK feature
    df["bhk"] = df["size"].apply(extract_bhk)

    # Remove rows where BHK could not be extracted
    df = df.dropna(subset=["bhk"])

    # Convert total_sqft into numeric
    df["total_sqft"] = df["total_sqft"].apply(
        convert_sqft_to_num
    )

    # Remove rows where sqft conversion failed
    df = df.dropna(subset=["total_sqft"])

    # Convert BHK to integer
    df["bhk"] = df["bhk"].astype(int)

    # Fill missing bathroom values using median
    df["bath"] = df["bath"].fillna(df["bath"].median())

    # Fill missing balcony values using median
    df["balcony"] = df["balcony"].fillna(
        df["balcony"].median()
    )

    # Society has many missing values.
    # We don't need it for our first model.
    df = df.drop(columns=["society"])

    # Original size column is no longer required
    df = df.drop(columns=["size"])

    # Availability is not required for the first model
    df = df.drop(columns=["availability"])

    # Area type is also not required for our first model
    df = df.drop(columns=["area_type"])

    # Keep only reasonable bathroom/BHK relationships
    df = df[df["bath"] <= df["bhk"] + 4]

    # Remove impossible/very small property areas
    df = df[df["total_sqft"] > 300]

    # Remove zero/negative prices
    df = df[df["price"] > 0]

    # Reset index
    df = df.reset_index(drop=True)

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned rows: {len(df)}")
    print(f"Cleaned columns: {list(df.columns)}")
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    clean_data()