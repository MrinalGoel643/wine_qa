# Wine Quality Prediction System

A full-stack machine learning system that predicts wine quality based on physicochemical properties. The project implements a reproducible ML pipeline with cloud deployment and an interactive web interface.

## Project Overview

This project builds an end-to-end wine quality prediction system. It sources data from cloud storage, trains a regression model to predict quality scores, deploys the model as an API endpoint, and provides a user-friendly frontend for making predictions.

## Dataset

We use the Wine Quality dataset containing physicochemical properties and quality ratings for red and white wines. The dataset includes 11 input features such as acidity, sugar content, pH levels, and alcohol percentage.

**Features:**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine type (red/white)

**Target Variable:** Quality score (0-10 scale)

## Model Architecture

The system uses a Random Forest Regressor trained on the wine quality dataset. Model predictions are categorized into quality levels:
- Poor: quality < 4
- Average: 4 ≤ quality < 6
- Good: 6 ≤ quality < 8
- Excellent: quality ≥ 8

Evaluation metrics include RMSE, MAE, and R² score tracked through MLflow experiment logging.

## Cloud Services Used

**Data Storage:** Azure Blob Storage  
**Model Deployment:** Azure Functions  
**API Framework:** Azure Functions HTTP Trigger  
**Frontend Hosting:** Streamlit Cloud  
**Experiment Tracking:** MLflow

## Repository Structure

```
├── data/
│   └── (training data splits)
├── models/
│   ├── wine_quality_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
├── api.py
├── app.py
├── function_app.py
├── train.py
├── preprocess_wine_data.py
├── download_from_azure.py
├── run_experiments.py
├── config.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── host.json
└── README.md
```

## Setup and Usage

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/MrinalGoel643/wine_qa.git
cd wine_qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# create .env file with your azure credentials
AZURE_STORAGE_CONNECTION_STRING=<your_connection_string>
AZURE_CONTAINER_NAME=wine-quality-data
```

4. Download data from Azure:
```bash
python download_from_azure.py
```

5. Train the model:
```bash
python train.py
```

6. Run the API locally:
```bash
python -m uvicorn api:app --reload
```

7. Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Expected response:
```json
{"predicted_quality": 5.12, "quality_category": "average"}
```

### Docker Deployment

Build and run the container:
```bash
docker-compose up
```

Or build manually:
```bash
docker build -t wine-quality-api .
docker run -p 8000:8000 wine-quality-api
```

### Cloud Deployment

The API is deployed on Azure Functions and accessible at:
```
https://wine-quality-ml.azurewebsites.net/api/predict
```

Available endpoints:
- `/api/predict` - POST request for wine quality prediction
- `/api/health` - GET request for health check
- `/api/debug` - GET request for debugging info

## Frontend Application

The web interface is built with Streamlit and provides an interactive way to input wine properties and receive quality predictions.

**Live Demo:** [Add Streamlit Cloud URL here after deployment]

Features:
- Adjustable sliders for all wine properties
- Real-time predictions from deployed API
- Quality categorization with visual indicators
- Input summary display

### Running Frontend Locally

```bash
streamlit run app.py
```

## Configuration

Model parameters and pipeline settings are controlled through `config.yaml`:
- Model hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- Train/test/validation split ratios
- Feature definitions (numerical and categorical)
- Data paths and storage configuration
- MLflow tracking settings

## Experiment Tracking

MLflow logs all training runs with:
- Model parameters
- Performance metrics (RMSE, MAE, R²)
- Trained model artifacts
- Preprocessing artifacts (scaler, encoder)

Access MLflow UI:
```bash
mlflow ui
```

View runs at `http://localhost:5000`

## Preprocessing Pipeline

The `preprocess_wine_data.py` script handles:
- Loading and combining red and white wine datasets
- Data cleaning (missing values, duplicates)
- Outlier removal using IQR method
- Train/validation/test splitting
- Target distribution analysis

## Training Pipeline

The `train.py` script implements:
- Data download from Azure Blob Storage
- Feature preprocessing (scaling numerical features, encoding categorical)
- Random Forest model training
- Evaluation on train/validation/test sets
- Model and artifact saving
- MLflow experiment tracking

## Hyperparameter Tuning

Run multiple experiments with different configurations:
```bash
python run_experiments.py
```

This runs experiments with varied hyperparameters and logs all results to MLflow.

## Collaboration

The project follows GitHub best practices:
- Feature branches for development
- Pull requests for code review
- Clear commit messages
- Comprehensive documentation

Each team member contributed through pull requests with code reviews before merging to main.

## Ethical Considerations

This model predicts wine quality based on chemical properties and should be used as a supplementary tool rather than the sole determinant of quality. Wine appreciation involves subjective elements beyond chemical composition. The model's predictions represent statistical patterns in the training data and may not capture all factors that contribute to wine quality.

## Limitations

- The model is trained on a specific dataset and may not generalize to all wine varieties
- Predictions are estimates based on physicochemical properties only
- Quality ratings in the training data reflect specific tasting panel preferences
- The model does not account for factors like vintage year, region, or storage conditions