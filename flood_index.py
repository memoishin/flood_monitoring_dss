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
    days_in_year = 365
    min_total_years = 10

    st.title('Daily Flood Index')

    st.write("Please upload the timeseries daily rainfall data. Once you upload the csv file, the application will start to compute the Flood Index.")
    st.write("After Flood Index is calculated, the results will be available for download.")
    st.write("The duration, severity and intensity of all levels of floods will also be computed by this application.")
    st.write("The minimum years of data required is 10. The two columns in the csv file should be named as 'date' and 'daily_rain'.")
    st.write("Antecedent Period of 365 days is considered during the computations.")
    st.write("If you will be running multiple computations, please refresh the webpage after you have downloaded the results for each.")

    antecedent_period = 365
    weight = 0
    for x in range (1, antecedent_period + 1, 1):
        weight = weight + 1/x

    st.sidebar.write("Weight (W) : ", weight)

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        start_date = df["date"][0]
        end_date = df["date"][len(df.index) - 1]
        st.write("Retrieved data from: ", start_date, " uptill", end_date)

        df['year'] = 0
        df['month'] = 0
        df['day'] = 0
        df['ep'] = 0.0
        df['awri'] = 0.0
        df['fi'] = 0.0


        st.sidebar.write("Deriving Day, Month and Year")
        latest_iteration = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        i = 0
        for index, row in df.iterrows():
            temp_date = datetime.strptime(row["date"], "%d/%m/%Y").date()
            df["year"][index] = temp_date.year
            df['month'][index] = temp_date.month
            df['day'][index] = temp_date.day
            i = i + 1
            latest_iteration.text(f'Iteration {i}')
            bar.progress((i/len(df.index)))

        
        st.sidebar.write("Making Adjustments for Leap Year")
        leap_indexes = []
        i = 0
        latest_iteration = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        for index, row in df.iterrows():
            i = i + 1
            latest_iteration.text(f'Iteration {i}')
            bar.progress((i/len(df.index)))
            
            if(df["day"][index] == 29 and df["month"][index] == 2):
                df["daily_rain"][index + 1] = df["daily_rain"][index] + df["daily_rain"][index + 1]
                leap_indexes.append(index)
        
        df = df.drop(leap_indexes) 
        df = df.reset_index(drop=True)
        
        total_years = int(len(df.index)/days_in_year)
        st.sidebar.write("Total Years of Data = ", total_years)

        if(total_years < 10):
            st.write("Atleast 10 Years of Data is Needed. Please upload data file that meets this requirement.")
        else:     
            st.sidebar.write("Calculating Missing Values")
            start_date = datetime.strptime(start_date, "%d/%m/%Y").date()
            current_year = start_date.year - 1
            raw_data = np.zeros(shape=(total_years, days_in_year))
            row = -1
            col = -1
            i = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for index, r in df.iterrows():
                i = i + 1
                latest_iteration.text(f'Iteration {i}')
                bar.progress((i/len(df.index)))
                if(df["year"][index] == current_year):
                    col = col + 1
                    if(math.isnan(df["daily_rain"][index])):
                        raw_data[row, col] = -1
                    else:
                        raw_data[row, col] = df["daily_rain"][index]
                else:
                    current_year = df["year"][index]
                    col = 0
                    row = row + 1
                    if(math.isnan(df["daily_rain"][index])):
                        raw_data[row, col] = -1
                    else:
                        raw_data[row, col] = df["daily_rain"][index]
            
            current_year = start_date.year - 1
            row = -1
            col = -1
            num_negatives = 0
            i = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for index, r in df.iterrows():
                i = i + 1
                latest_iteration.text(f'Iteration {i}')
                bar.progress((i/len(df.index)))
                if(df["year"][index] == current_year):
                    col = col + 1
                    if(math.isnan(df["daily_rain"][index])):
                        for j in range (total_years):
                            if(raw_data[j, col] == -1):
                                num_negatives = num_negatives + 1
                        df["daily_rain"][index] = (sum(raw_data[:, col]) + num_negatives)/(total_years-num_negatives)
                        num_negatives = 0
                else:
                    current_year = df["year"][index]
                    col = 0
                    row = row + 1
                    if(math.isnan(df["daily_rain"][index])):
                        for j in range (total_years):
                            if(raw_data[j, col] == -1):
                                num_negatives = num_negatives + 1
                        df["daily_rain"][index] = (sum(raw_data[:, col]) + num_negatives)/(total_years-num_negatives)
                        num_negatives = 0

            st.sidebar.write("Calculating Effective Precipitation")
            
            o = 0
            i = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            p = 0
            z = 0
            for i in range(antecedent_period, len(df.index), 1):
                o = o + 1
                latest_iteration.text(f'Iteration {o}')
                bar.progress((o/(len(df.index) - antecedent_period)))
                for j in range(i, i - antecedent_period, -1):
                    for k in range(i, j-1, -1):
                        p = p + df["daily_rain"][k]
                        z = z + 1
                    if(i <= len(df.index)):
                        df["ep"][i] = df["ep"][i] + p/z
                    p = 0
                    z = 0

            st.sidebar.write("Calculating Available Water Resource Index")
            
            o = 0
            latest_iteration = st.sidebar.empty()
            bar = st.sidebar.progress(0)
            for c in range(antecedent_period, len(df.index), 1):
                o = o + 1
                latest_iteration.text(f'Iteration {o}')
                bar.progress((o/(len(df.index) - antecedent_period)))
                df["awri"][c] = df["ep"][c] / weight

            st.sidebar.write("Calculating Flood Index")

            current_year = start_date.year + 1
            years_max = np.linspace(0, 0, total_years-1)
            yr = 0
            for y in range(antecedent_period, len(df.index), 1):
                if(df["year"][y] == current_year):
                    if(df["ep"][y] > years_max[yr]):
                        years_max[yr] = df["ep"][y]
                else:
                    yr = yr + 1
                    current_year = df["year"][y]
                    years_max[yr] = df["ep"][y]

            for x in range(antecedent_period, len(df.index), 1):
                df["fi"][x] = (df["ep"][x] - np.mean(years_max))/statistics.stdev(years_max)

            # Maximum Yearly Precipitation
            current_year = start_date.year + 1
            p_years_max = np.linspace(0, 0, total_years-1)
            yr = 0
            for y in range(antecedent_period, len(df.index), 1):
                if(df["year"][y] == current_year):
                    if(df["daily_rain"][y] > p_years_max[yr]):
                        p_years_max[yr] = df["daily_rain"][y]
                else:
                    yr = yr + 1
                    current_year = df["year"][y]
                    p_years_max[yr] = df["daily_rain"][y]

            # Peak Danger Yearly
            current_year = start_date.year + 1
            i_years_max = np.linspace(0, 0, total_years-1)
            yr = 0
            for y in range(antecedent_period, len(df.index), 1):
                if(df["year"][y] == current_year):
                    if(df["fi"][y] > i_years_max[yr]):
                        i_years_max[yr] = df["fi"][y]
                else:
                    yr = yr + 1
                    current_year = df["year"][y]
                    i_years_max[yr] = df["fi"][y]
    
            st.write("Flood Index Successfully Calculated")
            st.write(df)

            # Flood Index        
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)  

            plt.rcParams.update({'font.size': 15})
            res = df.drop(df.index[0:365])
            res = res.reset_index(drop=True)

            # Results Visualization

            st.write("Daily Flood Index and Rainfall Trend")

            fig, ax1 = plt.subplots(figsize=(10, 5))
            x_points = np.linspace(1, len(res.index), len(res.index))

            ax1.axhline(linewidth=1, color='r')
            ax1.axhline(y=1, linewidth=0.5, linestyle='--', color='grey')
            ax1.axhline(y=1.5, linewidth=0.5, linestyle='--', color='grey')
            ax1.axhline(y=2, linewidth=0.5, linestyle='--', color='grey')
            ax1.set_xlabel("Days since" + " " + res["date"][0])
            ax1.set_ylabel('Flood Index', color="black")
            ax1.plot(x_points, res['fi'], color="blue")
            ax1.tick_params(axis='y', labelcolor="black")

            ax2 = ax1.twinx()  

            ax2.set_ylabel('Precipitation (mm)', color="black")  
            ax2.bar(x_points, res['daily_rain'], color="grey")
            ax2.tick_params(axis='y', labelcolor="black")

            fig.tight_layout() 
            st.pyplot(fig)

            st.write("Daily AWRI and Rainfall Trend")

            fig2, ax3 = plt.subplots(figsize=(10, 5))
            x_points = np.linspace(1, len(res.index), len(res.index))

            ax3.set_xlabel("Days since" + " " + res["date"][0])
            ax3.set_ylabel('AWRI', color="black")
            ax3.plot(x_points, res['awri'], color="blue")
            ax3.tick_params(axis='y', labelcolor="black")

            ax4 = ax3.twinx()  

            ax4.set_ylabel('Precipitation (mm)', color="black")  
            ax4.bar(x_points, res['daily_rain'], color="grey")
            ax4.tick_params(axis='y', labelcolor="black")

            fig2.tight_layout() 
            st.pyplot(fig2)

            # Further Analysis

            st.sidebar.write("Computing Duration, Severity and Intensity of All Floods")

            # Identifying all floods
            flood = pd.DataFrame(columns=["onset", "end", "start_day", "start_month", "start_year", "end_day", "end_month", "end_year", "duration", "severity", "awri", "precipitation", "max_awri", "peak_severity"])
            num_floods = -1
            flood_state = False
            for y in range(antecedent_period, len(df.index), 1):
                if(df["fi"][y] > 0):
                    if(flood_state == False):
                        flood_state = True
                        num_floods = num_floods + 1
                        flood = flood.append({'onset': int(y), 'start_day': int(df['day'][y]), 'start_month': int(df['month'][y]), 'start_year': int(df['year'][y])}, ignore_index=True)
                
                if(df["fi"][y] <= 0):
                    if(flood_state == True):
                        flood_state = False
                        flood['end'][num_floods] = int(y - 1)
                        flood['end_day'][num_floods] = int(df['day'][y-1])
                        flood['end_month'][num_floods] = int(df['month'][y-1])
                        flood['end_year'][num_floods] = int(df['year'][y-1])
                        flood['duration'][num_floods] = int(y - flood['onset'][num_floods])

            t_severity = 0
            peak_sev = 0
            t_awri = 0
            t_prec = 0
            t_awri_max = 0

            for i in range(0, len(flood.index), 1):
                for j in range (int(flood['onset'][i]), int(flood['end'][i] + 1), 1):
                    t_severity = df['fi'][j] + t_severity
                    t_awri = df['awri'][j] + t_awri
                    t_prec = df['daily_rain'][j] + t_prec
                    
                    if(df['awri'][j] > t_awri_max):
                        t_awri_max = df['awri'][j]
                    
                    if(df['fi'][j] > peak_sev):
                        peak_sev = df['fi'][j]

                flood['severity'][i] = t_severity
                flood['awri'][i] = t_awri
                flood['precipitation'][i] = t_prec
                flood['max_awri'][i] = t_awri_max
                flood['peak_severity'][i] = peak_sev
                
                t_severity = 0
                peak_sev = 0
                t_awri = 0
                t_prec = 0
                t_awri_max = 0

            
            st.write("Duration, Severity and Intensity of All Floods Determined")
            st.write(flood)
            st.markdown(get_table_download_link(flood), unsafe_allow_html=True) 

            st.sidebar.write("Computing Duration, Severity and Intensity of Moderate Floods")

            # Identifying Moderate Floods
            flood = pd.DataFrame(columns=["onset", "end", "start_day", "start_month", "start_year", "end_day", "end_month", "end_year", "duration", "severity", "awri", "precipitation", "max_awri", "peak_severity"])
            num_floods = -1
            flood_state = False
            for y in range(antecedent_period, len(df.index), 1):
                if(df["fi"][y] >= 1):
                    if(flood_state == False):
                        flood_state = True
                        num_floods = num_floods + 1
                        flood = flood.append({'onset': int(y), 'start_day': int(df['day'][y]), 'start_month': int(df['month'][y]), 'start_year': int(df['year'][y])}, ignore_index=True)
                
                if(df["fi"][y] < 1):
                    if(flood_state == True):
                        flood_state = False
                        flood['end'][num_floods] = int(y - 1)
                        flood['end_day'][num_floods] = int(df['day'][y-1])
                        flood['end_month'][num_floods] = int(df['month'][y-1])
                        flood['end_year'][num_floods] = int(df['year'][y-1])
                        flood['duration'][num_floods] = int(y - flood['onset'][num_floods])

            t_severity = 0
            peak_sev = 0
            t_awri = 0
            t_prec = 0
            t_awri_max = 0

            for i in range(0, len(flood.index), 1):
                for j in range (int(flood['onset'][i]), int(flood['end'][i] + 1), 1):
                    t_severity = df['fi'][j] + t_severity
                    t_awri = df['awri'][j] + t_awri
                    t_prec = df['daily_rain'][j] + t_prec
                    
                    if(df['awri'][j] > t_awri_max):
                        t_awri_max = df['awri'][j]
                    
                    if(df['fi'][j] > peak_sev):
                        peak_sev = df['fi'][j]

                flood['severity'][i] = t_severity
                flood['awri'][i] = t_awri
                flood['precipitation'][i] = t_prec
                flood['max_awri'][i] = t_awri_max
                flood['peak_severity'][i] = peak_sev
                
                t_severity = 0
                peak_sev = 0
                t_awri = 0
                t_prec = 0
                t_awri_max = 0

            
            st.write("Duration, Severity and Intensity of Moderate Floods Determined")
            st.write(flood)
            st.markdown(get_table_download_link(flood), unsafe_allow_html=True)   

            # Identifying Severe floods

            st.sidebar.write("Computing Duration, Severity and Intensity of Severe Floods")

            flood = pd.DataFrame(columns=["onset", "end", "start_day", "start_month", "start_year", "end_day", "end_month", "end_year", "duration", "severity", "awri", "precipitation", "max_awri", "peak_severity"])
            num_floods = -1
            flood_state = False
            for y in range(antecedent_period, len(df.index), 1):
                if(df["fi"][y] >= 1.5):
                    if(flood_state == False):
                        flood_state = True
                        num_floods = num_floods + 1
                        flood = flood.append({'onset': int(y), 'start_day': int(df['day'][y]), 'start_month': int(df['month'][y]), 'start_year': int(df['year'][y])}, ignore_index=True)
                
                if(df["fi"][y] < 1.5):
                    if(flood_state == True):
                        flood_state = False
                        flood['end'][num_floods] = int(y - 1)
                        flood['end_day'][num_floods] = int(df['day'][y-1])
                        flood['end_month'][num_floods] = int(df['month'][y-1])
                        flood['end_year'][num_floods] = int(df['year'][y-1])
                        flood['duration'][num_floods] = int(y - flood['onset'][num_floods])

            t_severity = 0
            peak_sev = 0
            t_awri = 0
            t_prec = 0
            t_awri_max = 0

            for i in range(0, len(flood.index), 1):
                for j in range (int(flood['onset'][i]), int(flood['end'][i] + 1), 1):
                    t_severity = df['fi'][j] + t_severity
                    t_awri = df['awri'][j] + t_awri
                    t_prec = df['daily_rain'][j] + t_prec
                    
                    if(df['awri'][j] > t_awri_max):
                        t_awri_max = df['awri'][j]
                    
                    if(df['fi'][j] > peak_sev):
                        peak_sev = df['fi'][j]

                flood['severity'][i] = t_severity
                flood['awri'][i] = t_awri
                flood['precipitation'][i] = t_prec
                flood['max_awri'][i] = t_awri_max
                flood['peak_severity'][i] = peak_sev
                
                t_severity = 0
                peak_sev = 0
                t_awri = 0
                t_prec = 0
                t_awri_max = 0

            
            st.write("Duration, Severity and Intensity of Severe Floods Determined")
            st.write(flood)
            st.markdown(get_table_download_link(flood), unsafe_allow_html=True) 

            # Identifying Extreme floods
            
            st.sidebar.write("Computing Duration, Severity and Intensity of Extreme Floods")
            
            flood = pd.DataFrame(columns=["onset", "end", "start_day", "start_month", "start_year", "end_day", "end_month", "end_year", "duration", "severity", "awri", "precipitation", "max_awri", "peak_severity"])
            num_floods = -1
            flood_state = False
            for y in range(antecedent_period, len(df.index), 1):
                if(df["fi"][y] >= 2):
                    if(flood_state == False):
                        flood_state = True
                        num_floods = num_floods + 1
                        flood = flood.append({'onset': int(y), 'start_day': int(df['day'][y]), 'start_month': int(df['month'][y]), 'start_year': int(df['year'][y])}, ignore_index=True)
                
                if(df["fi"][y] < 2):
                    if(flood_state == True):
                        flood_state = False
                        flood['end'][num_floods] = int(y - 1)
                        flood['end_day'][num_floods] = int(df['day'][y-1])
                        flood['end_month'][num_floods] = int(df['month'][y-1])
                        flood['end_year'][num_floods] = int(df['year'][y-1])
                        flood['duration'][num_floods] = int(y - flood['onset'][num_floods])

            t_severity = 0
            peak_sev = 0
            t_awri = 0
            t_prec = 0
            t_awri_max = 0

            for i in range(0, len(flood.index), 1):
                for j in range (int(flood['onset'][i]), int(flood['end'][i] + 1), 1):
                    t_severity = df['fi'][j] + t_severity
                    t_awri = df['awri'][j] + t_awri
                    t_prec = df['daily_rain'][j] + t_prec
                    
                    if(df['awri'][j] > t_awri_max):
                        t_awri_max = df['awri'][j]
                    
                    if(df['fi'][j] > peak_sev):
                        peak_sev = df['fi'][j]

                flood['severity'][i] = t_severity
                flood['awri'][i] = t_awri
                flood['precipitation'][i] = t_prec
                flood['max_awri'][i] = t_awri_max
                flood['peak_severity'][i] = peak_sev
                
                t_severity = 0
                peak_sev = 0
                t_awri = 0
                t_prec = 0
                t_awri_max = 0

            
            st.write("Duration, Severity and Intensity of Extreme Floods Determined")
            st.write(flood)
            st.markdown(get_table_download_link(flood), unsafe_allow_html=True)