import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from glob import glob
import joblib

def load_data(data_dir):
    """Load all transaction datasets from CSV files and combine them"""
    try:
        # First, check if the directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Directory '{data_dir}' does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            return None
            
        # Get all CSV files in the data directory
        csv_files = glob(os.path.join(data_dir, '*.csv'))
        
        if not csv_files:
            print(f"Error: No CSV files found in {data_dir}")
            print(f"Directory contents: {os.listdir(data_dir)}")
            return None
        
        print(f"Found {len(csv_files)} CSV files")
        print("First few files:", csv_files[:5])
        
        # Load and combine all CSV files
        dfs = []
        for file_path in csv_files:
            try:
                print(f"\nLoading {file_path}...")
                df = pd.read_csv(file_path)
                if df is not None and not df.empty:
                    print(f"Successfully loaded {len(df)} rows")
                    print("Columns in this file:", df.columns.tolist())
                    dfs.append(df)
                else:
                    print(f"Warning: Empty or None data in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                print(f"File size: {os.path.getsize(file_path)} bytes")
        
        if not dfs:
            print("Error: No data was successfully loaded")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print("\nColumns in the final dataset:")
        print(combined_df.columns.tolist())
        
        # Check for required columns
        required_columns = ['TX_DATETIME', 'TX_AMOUNT', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            print(f"\nError: Missing required columns: {missing_columns}")
            return None
        
        return combined_df
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return None

def explore_data(df):
    """Explore and display basic information about the dataset"""
    print("\nDataset Information:")
    print(f"Number of transactions: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Plot fraud distribution
    plt.figure(figsize=(8, 6))
    df['TX_FRAUD'].value_counts().plot(kind='bar')
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Fraud (1) vs Non-Fraud (0)')
    plt.ylabel('Count')
    plt.show()
    
    # Plot transaction amount distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='TX_AMOUNT', hue='TX_FRAUD', bins=50)
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.show()

def preprocess_data(df):
    """Preprocess the data and create additional features"""
    if df is None:
        print("Error: No data to preprocess")
        return None
        
    try:
        # Check if TX_DATETIME column exists
        if 'TX_DATETIME' not in df.columns:
            print("Error: TX_DATETIME column not found in dataset")
            print("Available columns:", df.columns.tolist())
            return None
            
        # Convert datetime to datetime object if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['TX_DATETIME']):
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        
        # Extract time-based features
        df['HOUR'] = df['TX_DATETIME'].dt.hour
        df['DAY'] = df['TX_DATETIME'].dt.day
        df['MONTH'] = df['TX_DATETIME'].dt.month
        
        # Create amount-based features
        """ 
        taking log helps normalize the distribution 
        and makes the model more robust to extreme values and hence
        here we use log1p instead of log to avoid log(0) values.
        """
        df['AMOUNT_LOG'] = np.log1p(df['TX_AMOUNT'])  # using log1p to avoid log(0) values
        
        # Create customer-based features
        """ 
        we group the data by customer_id
        and then we calculate the mean, std and count
        of the transaction amount 
        """
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_AMOUNT': ['mean', 'std', 'count']
        }).reset_index()
        customer_stats.columns = ['CUSTOMER_ID', 'CUSTOMER_MEAN_AMOUNT', 'CUSTOMER_STD_AMOUNT', 'CUSTOMER_TX_COUNT']
        
        # Create terminal-based features
        # same goes for terminal_id similar to customer_id
        terminal_stats = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': ['mean', 'std', 'count']
        }).reset_index()
        terminal_stats.columns = ['TERMINAL_ID', 'TERMINAL_MEAN_AMOUNT', 'TERMINAL_STD_AMOUNT', 'TERMINAL_TX_COUNT']
        
        # Merge features back to main dataframe
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')
        
        return df
        
    except Exception as e:
        print(f"Error in preprocess_data: {str(e)}")
        return None

def train_model(X_train, y_train):
    """Train a Random Forest classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42) # using 100 trees for better performance
    model.fit(X_train, y_train) # training the model
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    y_pred = model.predict(X_test) # predicting the test set
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load the data
    print("Loading data...")
    data_dir = 'data/csv_files'  # Directory containing CSV files
    
    # error handling: if the data directory doesn't exist
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        print("Please make sure the data directory exists and contains CSV files.")
        return
    
    df = load_data(data_dir)
    if df is None:
        return
    
    # Explore the data
    print("\nExploring data...")
    explore_data(df)
    
    # Preprocess the data
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    
    # Save customer and terminal statistics for later use
    customer_stats = df.groupby('CUSTOMER_ID').agg({
        'TX_AMOUNT': ['mean', 'std', 'count']
    }).reset_index()
    customer_stats.columns = ['CUSTOMER_ID', 'CUSTOMER_MEAN_AMOUNT', 'CUSTOMER_STD_AMOUNT', 'CUSTOMER_TX_COUNT']
    customer_stats.set_index('CUSTOMER_ID', inplace=True)
    
    terminal_stats = df.groupby('TERMINAL_ID').agg({
        'TX_AMOUNT': ['mean', 'std', 'count']
    }).reset_index()
    terminal_stats.columns = ['TERMINAL_ID', 'TERMINAL_MEAN_AMOUNT', 'TERMINAL_STD_AMOUNT', 'TERMINAL_TX_COUNT']
    terminal_stats.set_index('TERMINAL_ID', inplace=True)
    
    # Save the statistics
    joblib.dump(customer_stats, 'customer_stats.pkl')
    joblib.dump(terminal_stats, 'terminal_stats.pkl')
    
    # Select features for training
    features = [
        'TX_AMOUNT', 'AMOUNT_LOG', 'HOUR', 'DAY', 'MONTH',
        'CUSTOMER_MEAN_AMOUNT', 'CUSTOMER_STD_AMOUNT', 'CUSTOMER_TX_COUNT',
        'TERMINAL_MEAN_AMOUNT', 'TERMINAL_STD_AMOUNT', 'TERMINAL_TX_COUNT'
    ]
    
    X = df[features]
    y = df['TX_FRAUD']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'fraud_detection_model.pkl')
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.show()

if __name__ == "__main__":
    main() 