import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/processed/features.csv"
MODEL_DIR = "models"


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


def train_models():

    print("Starting model training...")

    # Check dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Separate features and target
    X = df.drop(columns=["price"])
    y = df["price"]

    print(f"\nNumber of features: {X.shape[1]}")
    print(f"Target column: price")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Define models
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    results = {}

    # Train and evaluate models
    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        mae, rmse, r2 = evaluate_model(
            model,
            X_test,
            y_test
        )

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"{name} training completed.")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # Save trained model
        model_path = os.path.join(
            MODEL_DIR,
            f"{name}.pkl"
        )

        joblib.dump(model, model_path)

        print(f"Model saved: {model_path}")

    # Save feature columns
    feature_columns_path = os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )

    joblib.dump(
        list(X.columns),
        feature_columns_path
    )

    print(
        f"\nFeature columns saved: "
        f"{feature_columns_path}"
    )

    # Display comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(results).T

    print(results_df)

    # Save results
    results_path = os.path.join(
        MODEL_DIR,
        "model_results.csv"
    )

    results_df.to_csv(results_path)

    print(
        f"\nModel results saved: {results_path}"
    )

    print("\nModel training completed successfully.")


if __name__ == "__main__":
    train_models()