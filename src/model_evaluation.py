import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/processed/features.csv"
MODEL_DIR = "models"
OUTPUT_PATH = "models/evaluation_results.csv"


def evaluate_models():

    print("Starting model evaluation...")

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

    # Use the same split as model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTesting samples: {len(X_test)}")

    # Models to evaluate
    model_names = [
        "linear_regression",
        "decision_tree",
        "random_forest"
    ]

    evaluation_results = []

    for name in model_names:

        model_path = os.path.join(
            MODEL_DIR,
            f"{name}.pkl"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        print(f"\nEvaluating {name}...")

        # Load model
        model = joblib.load(model_path)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        evaluation_results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

    # Create results dataframe
    results_df = pd.DataFrame(
        evaluation_results
    )

    # Save results
    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print(
        f"\nEvaluation results saved to: "
        f"{OUTPUT_PATH}"
    )

    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    evaluate_models()