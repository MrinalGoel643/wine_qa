"""
Wine Quality Prediction API
FastAPI application to serve the trained Random Forest model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predict wine quality scores (0-10) based on chemical properties",
    version="1.0.0"
)

# Global variables for model artifacts
model = None
scaler = None
encoder = None
config = None


class WineFeatures(BaseModel):
    """Input features for wine quality prediction"""
    fixed_acidity: float = Field(..., description="Fixed acidity", ge=0, le=20)
    volatile_acidity: float = Field(..., description="Volatile acidity", ge=0, le=2)
    citric_acid: float = Field(..., description="Citric acid", ge=0, le=2)
    residual_sugar: float = Field(..., description="Residual sugar", ge=0, le=50)
    chlorides: float = Field(..., description="Chlorides", ge=0, le=1)
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide", ge=0, le=100)
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide", ge=0, le=300)
    density: float = Field(..., description="Density", ge=0.98, le=1.01)
    pH: float = Field(..., description="pH level", ge=2.5, le=4.5)
    sulphates: float = Field(..., description="Sulphates", ge=0, le=2)
    alcohol: float = Field(..., description="Alcohol percentage", ge=8, le=15)
    wine_type: str = Field(..., description="Wine type: 'red' or 'white'")

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
                "wine_type": "red"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_quality: float = Field(..., description="Predicted wine quality score (0-10)")
    quality_category: str = Field(..., description="Quality category: poor/average/good/excellent")


def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    global model, scaler, encoder, config
    
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        model_path = config['output']['model_path']
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = config['output']['scaler_path']
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load encoder
        encoder_path = config['output']['encoder_path']
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        
        print("✅ Model artifacts loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return False


def get_quality_category(score: float) -> str:
    """Convert numeric quality score to category"""
    if score < 4:
        return "poor"
    elif score < 6:
        return "average"
    elif score < 8:
        return "good"
    else:
        return "excellent"


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model_artifacts()
    if not success:
        print("Warning: Model artifacts not loaded. Train the model first!")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict wine quality",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_quality(features: WineFeatures):
    """
    Predict wine quality based on chemical properties
    
    Args:
        features: Wine chemical properties
    
    Returns:
        Predicted quality score and category
    """
    
    # Check if model is loaded
    if model is None or scaler is None or encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: python train.py"
        )
    
    try:
        # Prepare feature names in the correct order
        num_features = config['features']['numerical_features']
        cat_features = config['features']['categorical_features']
        
        # Create input dictionary with correct column names
        input_dict = {
            "fixed acidity": features.fixed_acidity,
            "volatile acidity": features.volatile_acidity,
            "citric acid": features.citric_acid,
            "residual sugar": features.residual_sugar,
            "chlorides": features.chlorides,
            "free sulfur dioxide": features.free_sulfur_dioxide,
            "total sulfur dioxide": features.total_sulfur_dioxide,
            "density": features.density,
            "pH": features.pH,
            "sulphates": features.sulphates,
            "alcohol": features.alcohol,
            "wine_type": features.wine_type
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical feature
        input_df['wine_type'] = encoder.transform(input_df['wine_type'])
        
        # Scale numerical features
        input_df[num_features] = scaler.transform(input_df[num_features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Clip prediction to valid range (0-10)
        prediction = np.clip(prediction, 0, 10)
        
        # Get quality category
        category = get_quality_category(prediction)
        
        return PredictionResponse(
            predicted_quality=round(float(prediction), 2),
            quality_category=category
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(wines: list[WineFeatures]):
    """
    Predict quality for multiple wines
    
    Args:
        wines: List of wine features
    
    Returns:
        List of predictions
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    predictions = []
    for wine in wines:
        pred = await predict_quality(wine)
        predictions.append(pred)
    
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Wine Quality Prediction API...")
    print("Make sure you've trained the model first: python train.py")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)