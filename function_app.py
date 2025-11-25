"""
Azure Functions App for Wine Quality Prediction
HTTP-triggered function for model serving
"""

import azure.functions as func
import logging
import json
import pickle
import pandas as pd
import numpy as np
import os

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Global variables for model artifacts
model = None
scaler = None
encoder = None

# Feature configuration (matches training)
NUMERICAL_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]
CATEGORICAL_FEATURES = ["wine_type"]


def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    global model, scaler, encoder
    
    if model is not None:
        logging.info("Models already loaded, skipping...")
        return True
    
    try:
        # Use absolute path
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        logging.info(f"Attempting to load models from: {models_dir}")
        logging.info(f"Current directory: {os.getcwd()}")
        logging.info(f"Models directory exists: {os.path.exists(models_dir)}")
        
        if os.path.exists(models_dir):
            logging.info(f"Files in models dir: {os.listdir(models_dir)}")
        
        # Load model
        model_path = os.path.join(models_dir, "wine_quality_model.pkl")
        logging.info(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        logging.info(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info("Scaler loaded")
        
        # Load encoder
        encoder_path = os.path.join(models_dir, "encoder.pkl")
        logging.info(f"Loading encoder from: {encoder_path}")
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        logging.info("Encoder loaded")
        
        logging.info("All model artifacts loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return False
    except Exception as e:
        logging.error(f"Error loading model artifacts: {e}")
        import traceback
        logging.error(traceback.format_exc())
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


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    logging.info('Health check endpoint triggered')
    
    # Try to load models if not loaded
    if model is None:
        logging.info("Models not loaded, attempting to load...")
        load_success = load_model_artifacts()
        logging.info(f"Load attempt result: {load_success}")
    
    model_loaded = model is not None
    
    response_data = {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None
    }
    
    logging.info(f"Health check response: {response_data}")
    
    return func.HttpResponse(
        json.dumps(response_data),
        mimetype="application/json",
        status_code=200
    )


@app.route(route="debug", methods=["GET"])
def debug(req: func.HttpRequest) -> func.HttpResponse:
    """Debug endpoint to check what files exist"""
    
    debug_info = {
        "cwd": os.getcwd(),
        "files_in_cwd": os.listdir(os.getcwd()),
        "script_file": __file__,
        "script_dir": os.path.dirname(__file__)
    }
    
    # Check for models directory
    models_path = "models"
    if os.path.exists(models_path):
        debug_info["models_exists"] = True
        debug_info["models_contents"] = os.listdir(models_path)
    else:
        debug_info["models_exists"] = False
    
    # Check script directory
    script_dir = os.path.dirname(__file__)
    debug_info["files_in_script_dir"] = os.listdir(script_dir)
    
    models_in_script = os.path.join(script_dir, "models")
    if os.path.exists(models_in_script):
        debug_info["models_in_script_dir"] = True
        debug_info["models_contents_script_dir"] = os.listdir(models_in_script)
        debug_info["models_full_path"] = models_in_script
    else:
        debug_info["models_in_script_dir"] = False
    
    # Check if models are loaded
    debug_info["model_loaded_in_memory"] = model is not None
    debug_info["scaler_loaded_in_memory"] = scaler is not None
    debug_info["encoder_loaded_in_memory"] = encoder is not None
    
    return func.HttpResponse(
        json.dumps(debug_info, indent=2),
        mimetype="application/json",
        status_code=200
    )


@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """
    Predict wine quality based on chemical properties
    
    Expected JSON body:
    {
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
    """
    logging.info('Prediction endpoint triggered')
    
    # Load model if not already loaded
    if model is None:
        logging.info("Model not loaded, loading now...")
        if not load_model_artifacts():
            return func.HttpResponse(
                json.dumps({
                    "error": "Model not loaded. Failed to load model files."
                }),
                mimetype="application/json",
                status_code=503
            )
    
    try:
        # Parse request body
        req_body = req.get_json()
        
        # Validate required fields
        required_fields = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
            "density", "pH", "sulphates", "alcohol", "wine_type"
        ]
        
        missing_fields = [field for field in required_fields if field not in req_body]
        if missing_fields:
            return func.HttpResponse(
                json.dumps({
                    "error": f"Missing required fields: {missing_fields}"
                }),
                mimetype="application/json",
                status_code=400
            )
        
        # Create input dictionary with correct column names
        input_dict = {
            "fixed acidity": req_body["fixed_acidity"],
            "volatile acidity": req_body["volatile_acidity"],
            "citric acid": req_body["citric_acid"],
            "residual sugar": req_body["residual_sugar"],
            "chlorides": req_body["chlorides"],
            "free sulfur dioxide": req_body["free_sulfur_dioxide"],
            "total sulfur dioxide": req_body["total_sulfur_dioxide"],
            "density": req_body["density"],
            "pH": req_body["pH"],
            "sulphates": req_body["sulphates"],
            "alcohol": req_body["alcohol"],
            "wine_type": req_body["wine_type"]
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical feature
        input_df['wine_type'] = encoder.transform(input_df['wine_type'])
        
        # Scale numerical features
        input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Clip prediction to valid range (0-10)
        prediction = np.clip(prediction, 0, 10)
        
        # Get quality category
        category = get_quality_category(prediction)
        
        # Return prediction
        return func.HttpResponse(
            json.dumps({
                "predicted_quality": round(float(prediction), 2),
                "quality_category": category
            }),
            mimetype="application/json",
            status_code=200
        )
        
    except ValueError as e:
        logging.error(f"Invalid request body: {e}")
        return func.HttpResponse(
            json.dumps({
                "error": "Invalid request body. Expected JSON."
            }),
            mimetype="application/json",
            status_code=400
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({
                "error": f"Prediction failed: {str(e)}"
            }),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="", methods=["GET"])
def root(req: func.HttpRequest) -> func.HttpResponse:
    """Root endpoint with API information"""
    logging.info('Root endpoint triggered')
    
    return func.HttpResponse(
        json.dumps({
            "message": "Wine Quality Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "/api/predict": "POST - Predict wine quality",
                "/api/health": "GET - Health check",
                "/api/debug": "GET - Debug info"
            }
        }),
        mimetype="application/json",
        status_code=200
    )