import streamlit as st
import pandas as pd
import numpy as np
from fraud_detection import load_data, preprocess_data, train_model
import joblib
import os
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Fraud Detection System")
st.markdown("""
This application helps detect potentially fraudulent transactions based on various patterns.
""")

# Sidebar for model training
with st.sidebar:
    st.header("Model Training")
    
    # Display directory status
    st.subheader("Directory Status")
    st.write(f"Current working directory: {os.getcwd()}")
    data_dir = 'data'
    if os.path.exists(data_dir):
        st.success(f"Data directory '{data_dir}' exists")
        st.write(f"Contents of data directory: {os.listdir(data_dir)}")
    else:
        st.error(f"Data directory '{data_dir}' does not exist!")
    
    # Training button and process
    if st.button("Train New Model"):
        with st.spinner("Training model..."):
            try:
                # Load data
                st.info("Loading data...")
                df = load_data('data')
                if df is None:
                    st.error("Failed to load data. Please check the console output for details.")
                    st.stop()
                
                st.success("Data loaded successfully!")
                st.write(f"Dataset shape: {df.shape}")
                st.write("Columns:", df.columns.tolist())
                
                # Preprocess data
                st.info("Preprocessing data...")
                df = preprocess_data(df)
                if df is None:
                    st.error("Failed to preprocess data. Please check the console output for details.")
                    st.stop()
                
                st.success("Data preprocessed successfully!")
                st.write(f"Processed dataset shape: {df.shape}")
                
                # Select features for training
                features = [
                    'TX_AMOUNT', 'AMOUNT_LOG', 'HOUR', 'DAY', 'MONTH',
                    'CUSTOMER_MEAN_AMOUNT', 'CUSTOMER_STD_AMOUNT', 'CUSTOMER_TX_COUNT',
                    'TERMINAL_MEAN_AMOUNT', 'TERMINAL_STD_AMOUNT', 'TERMINAL_TX_COUNT'
                ]
                
                # Verify all required features exist
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    st.error(f"Missing required features: {missing_features}")
                    st.stop()
                
                # Prepare data for training
                X = df[features]
                y = df['TX_FRAUD']
                
                # Train model
                st.info("Training model...")
                model = train_model(X, y)
                
                # Save model
                joblib.dump(model, 'fraud_detection_model.pkl')
                st.success("Model trained and saved successfully!")
                
                # Display statistics
                st.info("Dataset Statistics:")
                st.write(f"Total transactions: {len(df)}")
                st.write(f"Fraudulent transactions: {y.sum()}")
                st.write(f"Fraud rate: {(y.sum() / len(y) * 100):.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during model training: {str(e)}")
                st.stop()

# Main content area
st.header("Transaction Details")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    # Transaction details input
    st.subheader("Transaction Information")
    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    customer_id = st.text_input("Customer ID")
    terminal_id = st.text_input("Terminal ID")
    
    # Time selection
    transaction_time = st.time_input("Transaction Time")
    transaction_date = st.date_input("Transaction Date")

with col2:
    # Customer history input
    st.subheader("Customer History")
    customer_mean_amount = st.number_input("Customer Mean Amount", min_value=0.0, step=0.01)
    customer_std_amount = st.number_input("Customer Amount Standard Deviation", min_value=0.0, step=0.01)
    customer_tx_count = st.number_input("Customer Transaction Count", min_value=0, step=1)
    
    # Terminal history input
    st.subheader("Terminal History")
    terminal_mean_amount = st.number_input("Terminal Mean Amount", min_value=0.0, step=0.01)
    terminal_std_amount = st.number_input("Terminal Amount Standard Deviation", min_value=0.0, step=0.01)
    terminal_tx_count = st.number_input("Terminal Transaction Count", min_value=0, step=1)

# Fraud detection button
if st.button("Check for Fraud"):
    if not os.path.exists('fraud_detection_model.pkl'):
        st.error("Please train the model first using the sidebar!")
    else:
        # Load model
        model = joblib.load('fraud_detection_model.pkl')
        
        # Create feature vector
        features = {
            'TX_AMOUNT': amount,
            'AMOUNT_LOG': np.log1p(amount),  # log(1+x) transformation
            'HOUR': transaction_time.hour,
            'DAY': transaction_date.day,
            'MONTH': transaction_date.month,
            'CUSTOMER_MEAN_AMOUNT': customer_mean_amount,
            'CUSTOMER_STD_AMOUNT': customer_std_amount,
            'CUSTOMER_TX_COUNT': customer_tx_count,
            'TERMINAL_MEAN_AMOUNT': terminal_mean_amount,
            'TERMINAL_STD_AMOUNT': terminal_std_amount,
            'TERMINAL_TX_COUNT': terminal_tx_count
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of fraud
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction == 1:
            st.error(f"⚠️ This transaction is likely fraudulent!")
        else:
            st.success(f"✅ This transaction appears legitimate.")
        
        st.write(f"Fraud Probability: {probability:.2%}")
        
        # Display risk factors
        st.subheader("Risk Factors")
        
        # Amount risk
        if amount > 220:  # Threshold for high amount
            st.warning("⚠️ High transaction amount (>220)")
        
        # Customer behavior risk
        if amount > customer_mean_amount + 2 * customer_std_amount:
            st.warning("⚠️ Unusual amount for this customer")
        
        # Terminal behavior risk
        if amount > terminal_mean_amount + 2 * terminal_std_amount:
            st.warning("⚠️ Unusual amount for this terminal")

# Add some statistics and visualizations
st.header("Fraud Statistics")
if os.path.exists('fraud_detection_model.pkl'):
    model = joblib.load('fraud_detection_model.pkl')
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_names = [
        'Transaction Amount', 'Log Amount', 'Hour', 'Day', 'Month',
        'Customer Mean Amount', 'Customer Std Amount', 'Customer TX Count',
        'Terminal Mean Amount', 'Terminal Std Amount', 'Terminal TX Count'
    ]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature')) 