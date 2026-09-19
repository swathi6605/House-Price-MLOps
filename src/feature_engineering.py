import os
import pandas as pd


INPUT_PATH = "data/processed/cleaned_house_data.csv"
OUTPUT_PATH = "data/processed/features.csv"


def engineer_features():

    print("Starting feature engineering...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    # Load cleaned data
    df = pd.read_csv(INPUT_PATH)

    print(f"Input rows: {len(df)}")

    # -------------------------------------------------
    # 1. Group rare locations
    # -------------------------------------------------

    location_counts = df["location"].value_counts()

    # Locations with 10 or fewer records become "other"
    df["location"] = df["location"].apply(
        lambda x: x if location_counts[x] > 10 else "other"
    )

    print(
        f"Locations after grouping: "
        f"{df['location'].nunique()}"
    )

    # -------------------------------------------------
    # 2. Create price per square foot
    # -------------------------------------------------

    df["price_per_sqft"] = (
        df["price"] * 100000 / df["total_sqft"]
    )

    # -------------------------------------------------
    # 3. Remove unnecessary columns
    # -------------------------------------------------

    # price_per_sqft is useful for analysis,
    # but should NOT be used as an input feature
    # because it is calculated using the target price.

    df = df.drop(columns=["price_per_sqft"])

    # -------------------------------------------------
    # 4. One-hot encode location
    # -------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=["location"],
        dtype=int
    )

    # -------------------------------------------------
    # 5. Save feature dataset
    # -------------------------------------------------

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Feature dataset saved to: {OUTPUT_PATH}")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    engineer_features()