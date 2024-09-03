#_______ Library Import _______ #
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from datetime import datetime
import base64
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import markdown_functions as md


#_______ Page Setup _______ #
st.set_page_config(
    page_title="NASA Weather Exploration",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# _______ Functions _______ #
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


background_image = "images/background.jpg"
encoded_image = get_base64_of_bin_file(background_image)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown(md.get_navbar_markdown(), unsafe_allow_html=True)
st.markdown(md.background_and_tabs_styles(encoded_image), unsafe_allow_html = True)


# _______ Page _______ #
st.write("")
st.title("")
col_1, col_2, col_3 = st.columns([8, 2, 0.5])

with col_1:  
    st.title("Solar Flares")

with col_2:
    st.title("")
    st.image("images/logo_cs.png", width=200)

with col_3:
    st.write("")

st.write("Solar Flares")

# _______ Plots _______ #
daily_avg = pd.read_parquet('./data/df.parquet.gzip')
mars_weather_data = pd.read_parquet('./data/mars.parquet.gzip')
solar_flares_data = pd.read_parquet('./data/flare.parquet.gzip')
mars_weather_data = mars_weather_data[["terrestrial_date", "min_temp", "max_temp", "pressure"]]
earth_weather_data_daily = daily_avg[["date", "temperature_2m", "relative_humidity_2m", "rain", "direct_radiation_instant"]]
solar_flares_data = solar_flares_data[["peaktime", "classtype", "intensity"]]
# Rename Columns for Consistency:
mars_weather_data.rename(columns={"terrestrial_date": "date"}, inplace=True)
solar_flares_data.rename(columns={"peaktime": "date"}, inplace=True)

mars_weather_data['date'] = pd.to_datetime(mars_weather_data['date']).dt.date
earth_weather_data_daily['date'] = pd.to_datetime(earth_weather_data_daily['date']).dt.date
solar_flares_data['date'] = pd.to_datetime(solar_flares_data['date'], errors = 'coerce').dt.date
# Merge DataFrames:
merged_df = pd.merge(mars_weather_data, earth_weather_data_daily, on="date", how="left")
merged_df = pd.merge(merged_df, solar_flares_data, on="date", how="left")
merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
merged_df = merged_df.sort_values('date', ascending=False)
merged_df['solar_flare'] = merged_df['classtype'].notna().astype(int)
merged_df['intensity'] = merged_df['intensity'].fillna('None')
merged_df['classtype'] = merged_df['classtype'].fillna('None')

# Scale Data for Forecasting
scaler = MinMaxScaler()
merged_df[["min_temp", "max_temp", "pressure", "temperature_2m"]] = scaler.fit_transform(merged_df[["min_temp", "max_temp", "pressure", "temperature_2m"]])
earth_temp_daily_avg = merged_df.groupby('date')['temperature_2m'].mean().reset_index()
earth_temp_daily_avg['solar_flare'] = merged_df.groupby('date')['solar_flare'].max().reset_index()['solar_flare']
earth_temp_daily_avg['temperature_2m_smooth'] = earth_temp_daily_avg['temperature_2m'].rolling(window=7,center=True).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=earth_temp_daily_avg['date'],
    y=earth_temp_daily_avg['temperature_2m_smooth'],
    mode='lines',
    name='Earth Temperature'
))

fig.add_trace(go.Scatter(
    x=earth_temp_daily_avg['date'],
    y=earth_temp_daily_avg['temperature_2m_smooth'],
    mode='markers',
    marker=dict(
        color=earth_temp_daily_avg['solar_flare'], 
        size=10,
        showscale=True,
        colorscale='Viridis',
        colorbar=dict(
            title="Solar Flare Activity"
        )
    ),
    name='Solar Flare Activity'
))

fig.update_layout(
    title="Earth Temperature vs. Solar Flare Activity",
    xaxis_title="Date",
    yaxis_title="Earth Temperature (smoothed)",
    showlegend=True,
    legend_orientation="h",  # orientation to horizontal
    legend_x=0.5,  # center the legend horizontally
    legend_y=1.1,  # position the legend above the plot
    height=500, 
    width=1300,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='black')
)
st.plotly_chart(fig, theme=None)

