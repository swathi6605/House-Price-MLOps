import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================
# Project Paths
# ==============================

DATA_PATH = "data/processed/features.csv"
MODEL_DIR = "models"

# MLflow SQLite database
MLFLOW_DATABASE = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "House Price Prediction"


# ==============================
# MLflow Tracking
# ==============================

def train_and_log_models():

    print("Starting MLflow experiment tracking...")

    # Set MLflow tracking database
    mlflow.set_tracking_uri(
        MLFLOW_DATABASE
    )

    # Create or select experiment
    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow experiment: {EXPERIMENT_NAME}"
    )

    # ==============================
    # Check Dataset
    # ==============================

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    # ==============================
    # Load Dataset
    # ==============================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ==============================
    # Features and Target
    # ==============================

    X = df.drop(
        columns=["price"]
    )

    y = df["price"]

    print(
        f"\nNumber of features: {X.shape[1]}"
    )

    print(
        "Target column: price"
    )

    # ==============================
    # Train-Test Split
    # ==============================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ==============================
    # Model Configuration
    # ==============================

    model_configs = {

        "linear_regression": {
            "model_file": "linear_regression.pkl"
        },

        "decision_tree": {
            "model_file": "decision_tree.pkl"
        },

        "random_forest": {
            "model_file": "random_forest.pkl",
            "n_estimators": 100
        }
    }

    # ==============================
    # Train / Log Models
    # ==============================

    for model_name, config in model_configs.items():

        print(
            f"\nStarting MLflow run: {model_name}"
        )

        # Model path
        model_path = os.path.join(
            MODEL_DIR,
            config["model_file"]
        )

        # Check model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        # Load trained model
        model = joblib.load(
            model_path
        )

        # ==============================
        # Start MLflow Run
        # ==============================

        with mlflow.start_run(
            run_name=model_name
        ):

            # Predictions
            predictions = model.predict(
                X_test
            )

            # ==============================
            # Calculate Metrics
            # ==============================

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

            # ==============================
            # Log Parameters
            # ==============================

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "test_size",
                0.20
            )

            mlflow.log_param(
                "random_state",
                42
            )

            mlflow.log_param(
                "number_of_features",
                X.shape[1]
            )

            if model_name == "random_forest":

                mlflow.log_param(
                    "n_estimators",
                    config["n_estimators"]
                )

            # ==============================
            # Log Metrics
            # ==============================

            mlflow.log_metric(
                "MAE",
                mae
            )

            mlflow.log_metric(
                "RMSE",
                rmse
            )

            mlflow.log_metric(
                "R2",
                r2
            )

            # ==============================
            # Log Model
            # ==============================

            if model_name in [
                "decision_tree",
                "random_forest"
            ]:

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ]
                )

            else:

                mlflow.sklearn.log_model(
                    model,
                    name="model"
                )

            # ==============================
            # Display Results
            # ==============================

            print(
                f"Model: {model_name}"
            )

            print(
                f"MAE  : {mae:.4f}"
            )

            print(
                f"RMSE : {rmse:.4f}"
            )

            print(
                f"R2   : {r2:.4f}"
            )

            print(
                "Model logged to MLflow."
            )

            # Run ID
            run_id = mlflow.active_run().info.run_id

            print(
                f"Run ID: {run_id}"
            )

    # ==============================
    # Completed
    # ==============================

    print("\n" + "=" * 60)

    print(
        "MLFLOW TRACKING COMPLETED"
    )

    print("=" * 60)

    print(
        f"Experiment: {EXPERIMENT_NAME}"
    )

    print(
        "Tracking database: mlflow.db"
    )


# ==============================
# Main
# ==============================

if __name__ == "__main__":

    train_and_log_models()