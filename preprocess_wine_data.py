"""
Wine Quality Data Preprocessing Script - REGRESSION VERSION
Cleans and prepares wine quality data for regression ML pipeline
Predicts quality scores from 0-10
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_combine_data(red_wine_path, white_wine_path):
    """
    Load red and white wine datasets and combine them
    
    Args:
        red_wine_path: Path to red wine CSV
        white_wine_path: Path to white wine CSV
    
    Returns:
        Combined DataFrame
    """
    # Load datasets (assuming semicolon separator from UCI)
    print("Loading datasets...")
    red_wine = pd.read_csv(red_wine_path, sep=';')
    white_wine = pd.read_csv(white_wine_path, sep=';')
    
    # Add wine type indicator
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    # Combine
    wine_df = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    print(f"Loaded {len(red_wine)} red wines and {len(white_wine)} white wines")
    print(f"Total samples: {len(wine_df)}")
    
    return wine_df

def clean_data(df):
    """
    Clean the wine quality dataset
    
    Args:
        df: Raw wine DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    print("\n🧹 Cleaning data...")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Found {missing.sum()} missing values")
        print(missing[missing > 0])
        # Drop rows with missing values
        df = df.dropna()
    else:
        print("No missing values found")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows - removing them")
        df = df.drop_duplicates()
    else:
        print("No duplicates found")
    
    # Remove outliers (optional - using IQR method for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'quality']
    
    initial_len = len(df)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive filtering
        upper_bound = Q3 + 3 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    removed = initial_len - len(df)
    print(f"Removed {removed} outliers ({removed/initial_len*100:.1f}%)")
    
    return df

def analyze_target_distribution(df):
    """
    Analyze the quality score distribution for regression task
    
    Args:
        df: Wine DataFrame
    
    Returns:
        DataFrame (unchanged, just prints analysis)
    """
    print(f"Regression Target Analysis (Quality Scores 0-10):")
    print(f"\n   Quality Distribution:")
    quality_counts = df['quality'].value_counts().sort_index()
    for quality, count in quality_counts.items():
        percentage = count / len(df) * 100
        bar = '█' * int(percentage / 2)
        print(f"   Quality {quality}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    print(f"\n   Statistics:")
    print(f"   Mean quality: {df['quality'].mean():.2f}")
    print(f"   Median quality: {df['quality'].median():.1f}")
    print(f"   Std deviation: {df['quality'].std():.2f}")
    print(f"   Min quality: {df['quality'].min()}")
    print(f"   Max quality: {df['quality'].max()}")
    
    return df

def split_and_save_data(df, output_dir='data', test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train/val/test sets and save to CSV
    
    Args:
        df: Preprocessed DataFrame
        output_dir: Directory to save splits
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test split)
        random_state: Random seed for reproducibility
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True  # For regression, we just shuffle
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    # Save splits
    train.to_csv(f'{output_dir}/wine_train.csv', index=False)
    val.to_csv(f'{output_dir}/wine_val.csv', index=False)
    test.to_csv(f'{output_dir}/wine_test.csv', index=False)
    
    # Save full cleaned dataset
    df.to_csv(f'{output_dir}/wine_cleaned.csv', index=False)
    
    print(f"Data saved to '{output_dir}/' directory:")
    print(f"   Training set: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"   Validation set: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"   Test set: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    # Show mean quality in each split
    print(f"\n   Mean quality by split:")
    print(f"   Training:   {train['quality'].mean():.2f}")
    print(f"   Validation: {val['quality'].mean():.2f}")
    print(f"   Test:       {test['quality'].mean():.2f}")
    
    return train, val, test

def main():
    """Main preprocessing pipeline"""
    
    print("Wine Quality Data Preprocessing Pipeline - REGRESSION")
    print("=" * 60)
    
    # STEP 1: Load data
    # Update these paths to match your file locations
    red_wine_path = 'data/winequality-red.csv'
    white_wine_path = 'data/winequality-white.csv'
    
    df = load_and_combine_data(red_wine_path, white_wine_path)
    
    # STEP 2: Clean data
    df = clean_data(df)
    
    # STEP 3: Analyze target distribution
    df = analyze_target_distribution(df)
    
    # STEP 4: Display dataset info
    print("Final Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Show feature names
    feature_cols = [col for col in df.columns if col not in ['quality', 'wine_type']]
    print(f"\n   Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i}. {col}")
    print(f"   \n   Categorical: wine_type")
    print(f"   Target: quality (0-10)")
    
    print(f"Feature Statistics:")
    print(df.describe())
    
    # STEP 5: Split and save
    train, val, test = split_and_save_data(df)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("Task: REGRESSION - Predict quality scores (0-10)")
    print("Evaluation metrics to use: RMSE, MAE, R²")
    print("\nNext step: Upload cleaned data to Azure Blob Storage")

if __name__ == "__main__":
    main()