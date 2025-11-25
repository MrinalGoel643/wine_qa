"""
Wine Quality ML Training Pipeline
Trains a Random Forest regression model to predict wine quality scores
"""

import os
import yaml
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

from download_from_azure import download_wine_data

# Load environment variables
load_dotenv()


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_data_from_azure(config):
    """Download data from Azure Blob Storage"""
    print("\n" + "="*60)
    print("DOWNLOADING DATA FROM AZURE")
    print("="*60)
    
    container_name = config['data']['container_name']
    
    try:
        download_wine_data(
            container_name=container_name,
            output_dir="data"
        )
        print("Data downloaded successfully!")
    except Exception as e:
        print(f"Could not download from Azure: {e}")
        print("   Using local data if available...")


def load_data(config):
    """Load training, validation, and test data"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_df = pd.read_csv(config['data']['train_file'])
    val_df = pd.read_csv(config['data']['val_file'])
    test_df = pd.read_csv(config['data']['test_file'])
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def prepare_features(df, config, scaler=None, encoder=None, fit=False):
    """
    Prepare features for model training
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        scaler: StandardScaler (if None, creates new one)
        encoder: LabelEncoder (if None, creates new one)
        fit: Whether to fit the scaler/encoder
    
    Returns:
        X, y, scaler, encoder
    """
    target_col = config['features']['target_column']
    cat_features = config['features']['categorical_features']
    num_features = config['features']['numerical_features']
    
    # Separate features and target
    X = df[num_features + cat_features].copy()
    y = df[target_col].copy()
    
    # Encode categorical features
    if encoder is None:
        encoder = LabelEncoder()
    
    for cat_col in cat_features:
        if fit:
            X[cat_col] = encoder.fit_transform(X[cat_col])
        else:
            X[cat_col] = encoder.transform(X[cat_col])
    
    # Scale numerical features
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X[num_features] = scaler.fit_transform(X[num_features])
    else:
        X[num_features] = scaler.transform(X[num_features])
    
    return X, y, scaler, encoder


def create_model(config):
    """Create Random Forest model based on configuration"""
    model = RandomForestRegressor(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        min_samples_split=config['model']['min_samples_split'],
        min_samples_leaf=config['model']['min_samples_leaf'],
        random_state=config['model']['random_state'],
        n_jobs=-1
    )
    return model


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"{dataset_name} Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_artifacts(model, scaler, encoder, config):
    """Save model and preprocessing artifacts"""
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = config['output']['model_path']
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = config['output']['scaler_path']
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save encoder
    encoder_path = config['output']['encoder_path']
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Encoder saved to {encoder_path}")


def train_pipeline(config_path="config.yaml", download_data=True):
    """
    Main training pipeline
    
    Args:
        config_path: Path to configuration file
        download_data: Whether to download data from Azure
    """
    print("\n" + "="*60)
    print("WINE QUALITY ML TRAINING PIPELINE")
    print("="*60)
    
    # Load configuration
    config = load_config(config_path)
    print(f"Configuration loaded from {config_path}")
    print(f"   Model: Random Forest Regressor")
    print(f"   n_estimators: {config['model']['n_estimators']}")
    print(f"   max_depth: {config['model']['max_depth']}")
    
    # Download data from Azure
    if download_data:
        download_data_from_azure(config)
    
    # Load data
    train_df, val_df, test_df = load_data(config)
    
    # Set up MLFlow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Start MLFlow run
    with mlflow.start_run():
        print("\n" + "="*60)
        print("STARTING MLFLOW RUN")
        print("="*60)
        
        # Log configuration parameters
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_estimators", config['model']['n_estimators'])
        mlflow.log_param("max_depth", config['model']['max_depth'])
        mlflow.log_param("min_samples_split", config['model']['min_samples_split'])
        mlflow.log_param("min_samples_leaf", config['model']['min_samples_leaf'])
        mlflow.log_param("random_state", config['model']['random_state'])
        
        # Prepare features
        print("\n" + "="*60)
        print("PREPARING FEATURES")
        print("="*60)
        
        X_train, y_train, scaler, encoder = prepare_features(
            train_df, config, fit=True
        )
        X_val, y_val, _, _ = prepare_features(
            val_df, config, scaler=scaler, encoder=encoder, fit=False
        )
        X_test, y_test, _, _ = prepare_features(
            test_df, config, scaler=scaler, encoder=encoder, fit=False
        )
        
        print(f"Features prepared")
        print(f"   Training features shape: {X_train.shape}")
        print(f"   Number of features: {X_train.shape[1]}")
        
        # Create and train model
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        model = create_model(config)
        model.fit(X_train, y_train)
        
        print(f"Random Forest model trained successfully!")
        
        # Evaluate on all datasets
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        train_metrics = evaluate_model(model, X_train, y_train, "Training")
        val_metrics = evaluate_model(model, X_val, y_val, "Validation")
        test_metrics = evaluate_model(model, X_test, y_test, "Test")
        
        # Log metrics to MLFlow
        mlflow.log_metric("train_rmse", train_metrics['rmse'])
        mlflow.log_metric("train_mae", train_metrics['mae'])
        mlflow.log_metric("train_r2", train_metrics['r2'])
        
        mlflow.log_metric("val_rmse", val_metrics['rmse'])
        mlflow.log_metric("val_mae", val_metrics['mae'])
        mlflow.log_metric("val_r2", val_metrics['r2'])
        
        mlflow.log_metric("test_rmse", test_metrics['rmse'])
        mlflow.log_metric("test_mae", test_metrics['mae'])
        mlflow.log_metric("test_r2", test_metrics['r2'])
        
        # Save artifacts
        save_artifacts(model, scaler, encoder, config)
        
        # Log model to MLFlow
        mlflow.sklearn.log_model(model, "model")
        
        # Log config file
        mlflow.log_artifact(config_path)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Final Test Metrics:")
        print(f"   RMSE: {test_metrics['rmse']:.4f}")
        print(f"   MAE:  {test_metrics['mae']:.4f}")
        print(f"   R²:   {test_metrics['r2']:.4f}")
        print(f"Model saved to: {config['output']['model_path']}")
        print(f"MLFlow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"View MLFlow UI with: mlflow ui")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train wine quality prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading data from Azure (use local data)"
    )
    
    args = parser.parse_args()
    
    train_pipeline(
        config_path=args.config,
        download_data=not args.skip_download
    )