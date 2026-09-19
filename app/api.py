from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/random_forest.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(
        f"Feature columns not found at: {FEATURE_COLUMNS_PATH}"
    )


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


# --------------------------------------------------
# Create FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting Bengaluru house prices",
    version="1.0.0"
)


# --------------------------------------------------
# Input Data Model
# --------------------------------------------------

class HouseData(BaseModel):

    location: str = Field(
        ...,
        description="Location of the house"
    )

    total_sqft: float = Field(
        ...,
        gt=0,
        description="Total square feet"
    )

    bath: float = Field(
        ...,
        gt=0,
        description="Number of bathrooms"
    )

    balcony: float = Field(
        ...,
        ge=0,
        description="Number of balconies"
    )

    bhk: int = Field(
        ...,
        gt=0,
        description="Number of bedrooms"
    )


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "House Price Prediction API is running",
        "model": "Random Forest",
        "version": "1.0.0"
    }


# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_house_price(house: HouseData):

    # Convert input to dictionary
    input_data = house.model_dump()

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------
    # Location Encoding
    # --------------------------------------------------

    input_df["location"] = input_df["location"].apply(
        lambda x: x if x in feature_columns else "other"
    )

    # One-hot encode location
    input_df = pd.get_dummies(
        input_df,
        columns=["location"],
        dtype=int
    )

    # --------------------------------------------------
    # Add Missing Feature Columns
    # --------------------------------------------------

    for column in feature_columns:

        if column not in input_df.columns:
            input_df[column] = 0

    # Keep only training feature columns
    input_df = input_df[feature_columns]

    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price_lakhs": round(float(prediction), 2),
        "currency": "INR",
        "unit": "Lakhs"
    }