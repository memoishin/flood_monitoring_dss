import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math
import statistics
import base64

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
    st.title('Hourly Water Resource Index')

    st.write("Please upload the timeseries hourly rainfall data. Once you upload the csv file, the application will start to compute the Water Resource Index (WRI).")
    st.write("After WRI is calculated, the results will be available for download.")
    st.write("The WRI for 24, 48, 72, and 96 hours will be computed by this application.")
    st.write("The three columns in the csv file should be named as 'date', 'hour', and 'rain'.")
    st.write("Please ensure there are no missing values in the dataset.")
    st.write("If you will be running multiple computations, please refresh the webpage after you have downloaded the results for each.")
    st.sidebar.title('Computation Details')

    start_computation = False

    wri_uploaded_file = st.file_uploader("Choose a file")
    if wri_uploaded_file is not None:
        df = pd.read_csv(wri_uploaded_file)

        start_date = df["date"][0]
        end_date = df["date"][len(df.index) - 1]
        st.write("Retrieved data from: ", start_date, " uptill", end_date)

        df['year'] = 0
        df['month'] = 0
        df['day'] = 0
        df['wri_24'] = 0.0
        df['wri_48'] = 0.0
        df['wri_72'] = 0.0
        df['wri_96'] = 0.0

        start_date = st.date_input("Choose Computation Start Date", datetime.strptime(start_date, "%d/%m/%Y").date())
        end_date = st.date_input("Choose Computation End Date", datetime.strptime(end_date, "%d/%m/%Y").date())

        if st.button("Click Here to Start Computation using Selected Dates"):
            start_computation = True

        if(start_computation == True):

            st.sidebar.write("Deriving Day, Month and Year")
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            f = 0
            for index, row in df.iterrows():
                temp_date = datetime.strptime(row["date"], "%d/%m/%Y").date()
                df["year"][index] = temp_date.year
                df['month'][index] = temp_date.month
                df['day'][index] = temp_date.day
                f = f + 1
                latest_iteration.text(f'Iteration {f}')
                bar.progress((f/len(df.index)))

            start_date_index = df.index[(df["year"] == start_date.year) & (df["month"] == start_date.month) & (df["day"] == start_date.day) & (df["hour"] == 1)].tolist()
            end_date_index = df.index[(df["year"] == end_date.year) & (df["month"] == end_date.month) & (df["day"] == end_date.day) & (df["hour"] == 24)].tolist()

            df = df[start_date_index[0]:end_date_index[0] + 1]
            df = df.reset_index(drop=True)

            st.sidebar.write("Calculating 24 Hour WRI....")

            antecedent_period = 24
            wri = 0
            itr = 0
            last_itr = 0

            weight = 0
            for x in range (1, antecedent_period + 1, 1):
                weight = weight + 1/x

            new_weight = weight

            f = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for i in range(antecedent_period - 1, len(df.index), 1):
                f = f + 1
                latest_iteration.text(f'Iteration {f}')
                bar.progress((f/len(df.index)))

                df["wri_24"][i] = 0
                for j in range(i, i - antecedent_period, -1):
                    if(itr == 0):
                        wri = wri + df['rain'][j]
                    else:
                        new_weight = new_weight - 1/itr
                        wri = wri + ((df['rain'][j] * new_weight) / weight)
                        
                    itr = itr + 1
                df['wri_24'][i] = round(wri, 3)
                wri = 0
                itr = 0
                new_weight = weight

            st.sidebar.write("Calculating 48 Hour WRI....")

            antecedent_period = 48
            wri = 0
            itr = 0
            last_itr = 0

            weight = 0
            for x in range (1, antecedent_period + 1, 1):
                weight = weight + 1/x

            new_weight = weight

            f = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for i in range(antecedent_period - 1, len(df.index), 1):
                f = f + 1
                latest_iteration.text(f'Iteration {f}')
                bar.progress((f/len(df.index)))

                df["wri_48"][i] = 0
                for j in range(i, i - antecedent_period, -1):
                    if(itr == 0):
                        wri = wri + df['rain'][j]
                    else:
                        new_weight = new_weight - 1/itr
                        wri = wri + ((df['rain'][j] * new_weight) / weight)
                        
                    itr = itr + 1
                df['wri_48'][i] = round(wri, 3)
                wri = 0
                itr = 0
                new_weight = weight

            st.sidebar.write("Calculating 72 Hour WRI....")

            antecedent_period = 72
            wri = 0
            itr = 0
            last_itr = 0

            weight = 0
            for x in range (1, antecedent_period + 1, 1):
                weight = weight + 1/x

            new_weight = weight

            f = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for i in range(antecedent_period - 1, len(df.index), 1):
                f = f + 1
                latest_iteration.text(f'Iteration {f}')
                bar.progress((f/len(df.index)))

                df["wri_72"][i] = 0
                for j in range(i, i - antecedent_period, -1):
                    if(itr == 0):
                        wri = wri + df['rain'][j]
                    else:
                        new_weight = new_weight - 1/itr
                        wri = wri + ((df['rain'][j] * new_weight) / weight)
                        
                    itr = itr + 1
                df['wri_72'][i] = round(wri, 3)
                wri = 0
                itr = 0
                new_weight = weight

            st.sidebar.write("Calculating 96 Hour WRI....")

            antecedent_period = 96
            wri = 0
            itr = 0
            last_itr = 0

            weight = 0
            for x in range (1, antecedent_period + 1, 1):
                weight = weight + 1/x

            new_weight = weight

            f = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for i in range(antecedent_period - 1, len(df.index), 1):
                f = f + 1
                latest_iteration.text(f'Iteration {f}')
                bar.progress((f/len(df.index)))

                df["wri_96"][i] = 0
                for j in range(i, i - antecedent_period, -1):
                    if(itr == 0):
                        wri = wri + df['rain'][j]
                    else:
                        new_weight = new_weight - 1/itr
                        wri = wri + ((df['rain'][j] * new_weight) / weight)
                        
                    itr = itr + 1
                df['wri_96'][i] = round(wri, 3)
                wri = 0
                itr = 0
                new_weight = weight

            st.write("Computations Completed")
            st.write(df)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            # Results Visualization
            plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})
            res = df.drop(df.index[0:96])
            res = res.reset_index(drop=True)

            st.write("WRI Trends - Interactive Graph")
                
            ig_data = pd.DataFrame(np.column_stack((res['wri_24'], res['wri_48'], res['wri_72'], res['wri_96'],np.zeros(len(res)))), columns=['24 Hr WRI', '48 Hr WRI', '72 Hr WRI', '96 Hr WRI', ''])
            st.line_chart(ig_data)

            st.write("24 Hour WRI Trend")
            
            fig, ax1 = plt.subplots(figsize=(15, 5))
            x_points = np.linspace(1, len(res.index), len(res.index))
            ax1.set_xlabel("Hours since" + " " + res["date"][0])
            ax1.set_ylabel('24 Hour WRI', color="black")
            ax1.plot(x_points, res['wri_24'], color="midnightblue")
            ax1.tick_params(axis='y', labelcolor="black")

            fig.tight_layout() 
            st.pyplot(fig)

            st.write("48 Hour WRI Trend")
            fig2, ax2 = plt.subplots(figsize=(15, 5))
            x_points = np.linspace(1, len(res.index), len(res.index))
            ax2.set_xlabel("Hours since" + " " + res["date"][0])
            ax2.set_ylabel('48 Hour WRI', color="black")
            ax2.plot(x_points, res['wri_48'], color="teal")
            ax2.tick_params(axis='y', labelcolor="black")

            fig2.tight_layout() 
            st.pyplot(fig2)

            st.write("72 Hour WRI Trend")
            fig3, ax3 = plt.subplots(figsize=(15, 5))
            x_points = np.linspace(1, len(res.index), len(res.index))
            ax3.set_xlabel("Hours since" + " " + res["date"][0])
            ax3.set_ylabel('72 Hour WRI', color="black")
            ax3.plot(x_points, res['wri_72'], color="firebrick")
            ax3.tick_params(axis='y', labelcolor="black")

            fig3.tight_layout() 
            st.pyplot(fig3)

            st.write("96 Hour WRI Trend")
            fig4, ax4 = plt.subplots(figsize=(15, 5))
            x_points = np.linspace(1, len(res.index), len(res.index))
            ax4.set_xlabel("Hours since" + " " + res["date"][0])
            ax4.set_ylabel('96 Hour WRI', color="black")
            ax4.plot(x_points, res['wri_96'], color="darkmagenta")
            ax4.tick_params(axis='y', labelcolor="black")

            fig4.tight_layout() 
            st.pyplot(fig4)