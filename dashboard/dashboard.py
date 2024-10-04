import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Memuat dataset
st.title("Dashboard Analisis Kualitas Udara")

# Memuat data
@st.cache_data
def load_data():
    df = pd.read_csv("PRSA_Data_20130301-20170228/PRSA_Data_Nongzhanguan_20130301-20170228.csv")
    df = df.infer_objects(copy=False)
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Senin = 0, Minggu = 6
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")
pollutant = st.sidebar.selectbox("Pilih Polutan", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
day_of_week = st.sidebar.multiselect("Pilih Hari dalam Seminggu", options=[0, 1, 2, 3, 4, 5, 6], default=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x])

filtered_df = df[df['day_of_week'].isin(day_of_week)]

# Visualisasi per jam
st.header(f"Rata-rata Level {pollutant} per Jam dalam Sehari")
hourly_pollution = filtered_df.groupby('hour')[pollutant].mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hourly_pollution.index, hourly_pollution, marker='o', label=pollutant)
ax.set_title(f'Rata-rata Level {pollutant} per Jam dalam Sehari')
ax.set_xlabel('Jam dalam Sehari')
ax.set_ylabel(f'Konsentrasi {pollutant}')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Visualisasi per hari dalam seminggu
st.header(f"Rata-rata Level {pollutant} per Hari dalam Seminggu")
weekly_pollution = filtered_df.groupby('day_of_week')[pollutant].mean()
hari_dict = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(weekly_pollution.index, weekly_pollution, marker='o', label=pollutant)
ax.set_title(f'Rata-rata Level {pollutant} per Hari dalam Seminggu')
ax.set_xlabel('Hari dalam Seminggu')
ax.set_xticks(ticks=weekly_pollution.index)
ax.set_xticklabels(labels=[hari_dict[i] for i in weekly_pollution.index])
ax.set_ylabel(f'Konsentrasi {pollutant}')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Matriks korelasi antara kecepatan angin dan polutan
st.header("Korelasi Antara Kecepatan Angin (WSPM) dan Polutan")
wind_pollution_corr = df[['WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(wind_pollution_corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Korelasi Antara Kecepatan Angin (WSPM) dan Polutan')
st.pyplot(fig)

# Scatter plot untuk hubungan antara WSPM dan PM2.5
st.header("Scatter Plot: Kecepatan Angin (WSPM) vs Konsentrasi PM2.5")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['WSPM'], df['PM2.5'], alpha=0.5)
ax.set_title('Scatter Plot: Kecepatan Angin (WSPM) vs Konsentrasi PM2.5')
ax.set_xlabel('Kecepatan Angin (m/s)')
ax.set_ylabel('Konsentrasi PM2.5')
ax.grid(True)
st.pyplot(fig)