import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math
import statistics
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, RepeatVector, ConvLSTM2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import csv
from scipy.stats import pearsonr
import HydroErr as he
import time

# split a multivariate sequence into samples
def split_sequences_msmv(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def get_table_download_link(dafr):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = dafr.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="Results.csv">Download Result (CSV)</a>'

def app():
    st.title('Hourly Flood Forecasting (Demo)')

    st.write("This section demonstrates the forecasting accuracy of the Artifical Intelligence Based forecasting method developed. Please enter the forecast horizon and lagged hours and then upload the data file. This can be the results file obtained from the 'Hourly Water Resources Index' section of this application.")

    st.sidebar.title('Model Configuration')
    test_s = st.sidebar.number_input('Testing Split Ratio', min_value = 0.01, max_value = 0.99, value=0.1)
    valid_s = st.sidebar.number_input('Validation Split Ratio (From Training Data)', min_value = 0.01, max_value = 0.99, value=0.2)
    filter_s = st.sidebar.number_input('ConvLSTM Filters (Layer 1)', min_value = 8, max_value = 512, value=256)
    filter_s2 = st.sidebar.number_input('ConvLSTM Filters (Layer 2)', min_value = 8, max_value = 512, value=128)
    filter_s3 = st.sidebar.number_input('ConvLSTM Filters (Layer 3)', min_value = 8, max_value = 512, value=64)
    epochs_s = st.sidebar.number_input('Epochs', min_value = 1, max_value = 1000, value=75)
    batch_s = st.sidebar.number_input('Batch Size', min_value = 1, max_value = 1000, value=100)

    st.sidebar.title('Computation Details')

    horizon = st.number_input('Enter the forecast horizon (Hours)', min_value = 1, max_value = 6)
    lags = st.number_input('Enter the number of lagged hours to consider', min_value = 3, max_value = 14)

    uploaded_file = st.file_uploader("Please upload time series WRI. 96 Hour WRI is used for forecasting. The column should be labelled as 'wri_96'.")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        data = df[95:]
        data = data.reset_index(drop=True)

        raw_seq = data
        raw_seq['wri_96_lag_1'] = raw_seq['wri_96'].shift(1)
        # raw_seq['rain_lag_1'] = raw_seq['rain'].shift(1)
        raw_seq = raw_seq.iloc[1:]
        raw_seq

        train, test = train_test_split(raw_seq, test_size=test_s, random_state=None, shuffle=False)

        # in_seq1= train['rain_lag_1']
        in_seq2= train['wri_96_lag_1']
        out_seq = train['wri_96']

        # convert to [rows, columns] structure
        # in_seq1 = in_seq1.values.reshape((len(in_seq1), 1))
        in_seq2 = in_seq2.values.reshape((len(in_seq2), 1))
        out_seq = out_seq.values.reshape((len(out_seq), 1))

        # horizontally stack columns
        dataset = np.hstack((in_seq2, out_seq))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(dataset)

        # choose a number of time steps
        n_steps_in, n_steps_out = lags, horizon

        # covert into input/output
        X, y = split_sequences_msmv(scaled, n_steps_in, n_steps_out)
        # the dataset knows the number of features, e.g. 2
        n_features = X.shape[2]
        n_seq = 1
        n_steps = lags
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

        st.sidebar.write("Model Running")
        tic = time.perf_counter()
        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=filter_s, kernel_size=(1,2), activation='relu', return_sequences=True,  input_shape=(n_seq, 1, n_steps, n_features)))
        model.add(ConvLSTM2D(filters=filter_s2, kernel_size=(1,2), activation='relu', return_sequences=True))
        model.add(ConvLSTM2D(filters=filter_s3, kernel_size=(1), activation='relu', return_sequences=False))
        # model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model_history = model.fit(X, y, epochs=epochs_s, validation_split=valid_s, batch_size=batch_s, verbose=0)
        toc = time.perf_counter()
        
        st.sidebar.write("Model Running Completed in ", round(toc - tic, 2), " seconds")

        plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})

        st.markdown("**Visualizing Loss during Training and Validation**")
        
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        ax2.plot(model_history.history['loss'], 'b', label='Training Loss')
        ax2.plot(model_history.history['val_loss'], 'r', label='Validation Loss')
        ax2.set_title('Model Loss During Training and Validation')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_xlabel('Epochs')
        ax2.legend(loc='upper right')
        plt.show()
        fig2.tight_layout() 
        st.pyplot(fig2)

        # tin_seq1= train['rain_lag_1']
        tin_seq2= test['wri_96_lag_1']
        tout_seq = test['wri_96']

        # convert to [rows, columns] structure

        # tin_seq1 = tin_seq1.values.reshape((len(tin_seq1), 1))
        tin_seq2 = tin_seq2.values.reshape((len(tin_seq2), 1))
        tout_seq = tout_seq.values.reshape((len(tout_seq), 1))

        scaler_fi = MinMaxScaler(feature_range=(0, 1))
        scaled_fi = scaler_fi.fit_transform(tout_seq)

        # horizontally stack columns
        t_dataset = np.hstack((tin_seq2, tout_seq))

        t_scaled = scaler.fit_transform(t_dataset)
        X_test, y_test = split_sequences_msmv(t_scaled, n_steps_in, n_steps_out)

        X_test = X_test.reshape((X_test.shape[0], n_seq, 1, n_steps, n_features))

        predictions = model.predict(X_test, verbose=0)

        predictions_ns = scaler_fi.inverse_transform(predictions)

        y_test_ns = scaler_fi.inverse_transform(y_test)

        st.markdown("**Performance Evaluation Using Testing Data**")

        st.write("Visualization: Actual (Blue) vs Forecasted (Red)")
        
        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(y_test_ns, 'b', label='Actual')
        ax1.plot(predictions_ns, 'r', label='Forecasted')
        ax1.set_ylabel('WRI')
        ax1.set_xlabel('Hours')
        # ax1.legend()
        fig.tight_layout() 
        st.pyplot(fig)

        st.markdown("**Results from Statistical Performance Metrics**")
        r, _ = pearsonr(y_test_ns.flatten(), predictions_ns.flatten())
        rmse = he.rmse(predictions_ns.flatten(), y_test_ns.flatten())
        mae = he.mae(predictions_ns.flatten(), y_test_ns.flatten())
        willmott = he.d(predictions_ns.flatten(), y_test_ns.flatten())
        nash = he.nse(predictions_ns.flatten(), y_test_ns.flatten())
        leg = he.lm_index(predictions_ns.flatten(), y_test_ns.flatten())

        st.write("Pearson Correlation Coefficient: ", r.round(5))
        st.write("Legate McCabe Efficiency Index: ", leg.round(5))
        st.write("Willmott's Index: ", willmott.round(5))
        st.write("Nash Sutcliffe Efficiency Index: ", nash.round(5))
        st.write("Root Mean Squared Error: ", rmse.round(5))
        st.write("Mean Absolute Error: ", mae.round(5))
