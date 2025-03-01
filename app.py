import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import io

# Streamlit UI enhancements
st.set_page_config(page_title="StockistAI", layout="wide")

# Sidebar for user inputs
st.sidebar.title("StockistAI: Hybrid Stock Trend Forecasting")
st.sidebar.markdown("## Enter Stock Details")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
time_step = st.sidebar.slider("Time Step for LSTM:", 10, 100, 50)
epochs = st.sidebar.slider("Training Epochs:", 5, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size:", [16, 32, 64], index=1)
kernel_type = st.sidebar.selectbox("SVM Kernel:", ["linear", "rbf", "poly"], index=1)

def load_data(ticker):
    stock = yf.download(ticker, period="5y", interval="1d").dropna()
    return stock

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)
    return scaled_data, scaler

def create_sequences(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        Y.append(data[i+time_step])
    return np.array(X), np.array(Y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_svm(X_train, Y_train, kernel):
    svm_model = SVR(kernel=kernel)
    svm_model.fit(X_train, Y_train)
    return svm_model

def main():
    st.title("📈 StockistAI: Hybrid Stock Forecasting Model")
    st.markdown("### Predict future stock prices using LSTM & SVM models")
    
    if st.sidebar.button("Run Prediction"):
        with st.spinner("Fetching stock data..."):
            data = load_data(ticker)
        st.success("Stock data loaded successfully!")
        st.line_chart(data['Close'])
        
        with st.spinner("Preprocessing data..."):
            scaled_data, scaler = preprocess_data(data)
            X, Y = create_sequences(scaled_data, time_step)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        st.success("Data preprocessed successfully!")
        
        with st.spinner("Training LSTM model..."):
            lstm_model = build_lstm_model((X_train.shape[1], 1))
            lstm_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            lstm_predictions = lstm_model.predict(X_test)
        st.success("LSTM Model trained successfully!")
        
        with st.spinner("Training SVM model..."):
            X_svm_train = X_train.reshape(X_train.shape[0], -1)
            X_svm_test = X_test.reshape(X_test.shape[0], -1)
            svm_model = train_svm(X_svm_train, Y_train, kernel_type)
            svm_predictions = svm_model.predict(X_svm_test)
        st.success("SVM Model trained successfully!")
        
        hybrid_predictions = (lstm_predictions.flatten() + svm_predictions) / 2
        hybrid_predictions = scaler.inverse_transform(hybrid_predictions.reshape(-1, 1))
        Y_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(Y_actual, hybrid_predictions)
        mse = mean_squared_error(Y_actual, hybrid_predictions)
        st.sidebar.write(f"📊 MAE: {mae:.2f}")
        st.sidebar.write(f"📉 MSE: {mse:.2f}")
        
        st.subheader("Stock Price Prediction vs Actual")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(Y_actual, label='Actual Price', color='blue')
        ax.plot(hybrid_predictions, label='Predicted Price', color='red')
        ax.legend()
        st.pyplot(fig)
        
        # Prepare CSV for download
        df_results = pd.DataFrame({"Actual Price": Y_actual.flatten(), "Predicted Price": hybrid_predictions.flatten()})
        csv = df_results.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="stock_predictions.csv">📥 Download Predictions CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
