'''
This script was used to test a single transaction
'''

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def preprocess_transaction(transaction, customer_stats, terminal_stats):
    """Preprocess a single transaction with the same features used in training"""
    # Convert datetime string to datetime object
    tx_datetime = pd.to_datetime(transaction['TX_DATETIME'])
    
    # Create time-based features
    features = {
        'TX_AMOUNT': transaction['TX_AMOUNT'],
        'AMOUNT_LOG': np.log1p(transaction['TX_AMOUNT']),
        'HOUR': tx_datetime.hour,
        'DAY': tx_datetime.day,
        'MONTH': tx_datetime.month
    }
    
    # Add customer statistics
    customer_id = transaction['CUSTOMER_ID']
    if customer_id in customer_stats.index:
        features.update({
            'CUSTOMER_MEAN_AMOUNT': customer_stats.loc[customer_id, 'CUSTOMER_MEAN_AMOUNT'],
            'CUSTOMER_STD_AMOUNT': customer_stats.loc[customer_id, 'CUSTOMER_STD_AMOUNT'],
            'CUSTOMER_TX_COUNT': customer_stats.loc[customer_id, 'CUSTOMER_TX_COUNT']
        })
    else:
        # If customer is new, use default values
        features.update({
            'CUSTOMER_MEAN_AMOUNT': 0,
            'CUSTOMER_STD_AMOUNT': 0,
            'CUSTOMER_TX_COUNT': 0
        })
    
    # Add terminal statistics
    terminal_id = transaction['TERMINAL_ID']
    if terminal_id in terminal_stats.index:
        features.update({
            'TERMINAL_MEAN_AMOUNT': terminal_stats.loc[terminal_id, 'TERMINAL_MEAN_AMOUNT'],
            'TERMINAL_STD_AMOUNT': terminal_stats.loc[terminal_id, 'TERMINAL_STD_AMOUNT'],
            'TERMINAL_TX_COUNT': terminal_stats.loc[terminal_id, 'TERMINAL_TX_COUNT']
        })
    else:
        # If terminal is new, use default values
        features.update({
            'TERMINAL_MEAN_AMOUNT': 0,
            'TERMINAL_STD_AMOUNT': 0,
            'TERMINAL_TX_COUNT': 0
        })
    
    return pd.DataFrame([features])

def test_transaction():
    """Test a single transaction with the trained model"""
    try:
        # Load the trained model and statistics
        model = joblib.load('fraud_detection_model.pkl')
        customer_stats = joblib.load('customer_stats.pkl')
        terminal_stats = joblib.load('terminal_stats.pkl')
        
        # Example transaction - you can modify these values to test different scenarios
        transaction = {
            'TX_DATETIME': '2018-04-15 12:00:00',
            'CUSTOMER_ID': 4693,  # Example customer ID
            'TERMINAL_ID': 5040,  # Example terminal ID
            'TX_AMOUNT': 1000.00  # Example amount
        }
        
        # Preprocess the transaction
        features = preprocess_transaction(transaction, customer_stats, terminal_stats)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Probability of being fraudulent
        
        # Print results
        print("\nTransaction Details:")
        print(f"Time: {transaction['TX_DATETIME']}")
        print(f"Customer ID: {transaction['CUSTOMER_ID']}")
        print(f"Terminal ID: {transaction['TERMINAL_ID']}")
        print(f"Amount: ${transaction['TX_AMOUNT']:.2f}")
        print(f"\nPrediction: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'}")
        print(f"Fraud Probability: {probability:.2%}")
        
        # Print feature values
        print("\nFeature Values:")
        for feature, value in features.iloc[0].items():
            print(f"{feature}: {value:.2f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have run the training script first to generate the model and statistics files.")

if __name__ == "__main__":
    test_transaction() 
