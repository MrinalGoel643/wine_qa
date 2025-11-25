"""
Run multiple experiments with different hyperparameters
Useful for hyperparameter tuning
"""

import yaml
import os
from train import train_pipeline


def run_experiments():
    """Run multiple training experiments with different Random Forest configurations"""
    
    # Define experiments with different hyperparameters
    experiments = [
        {
            "name": "baseline",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        },
        {
            "name": "deep_forest",
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        },
        {
            "name": "shallow_forest",
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5
        },
        {
            "name": "large_forest",
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }
    ]
    
    # Load base config
    with open("config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Run each experiment
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {exp['name']}")
        print(f"{'='*60}")
        
        # Modify config
        config = base_config.copy()
        config['model']['n_estimators'] = exp['n_estimators']
        config['model']['max_depth'] = exp['max_depth']
        config['model']['min_samples_split'] = exp['min_samples_split']
        config['model']['min_samples_leaf'] = exp['min_samples_leaf']
        
        # Save temporary config
        temp_config_path = f"config_{exp['name']}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run training
        try:
            train_pipeline(
                config_path=temp_config_path,
                download_data=(i == 0)  # Only download on first run
            )
        except Exception as e:
            print(f"Experiment {exp['name']} failed: {e}")
        finally:
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print("View results with: mlflow ui")


if __name__ == "__main__":
    run_experiments()