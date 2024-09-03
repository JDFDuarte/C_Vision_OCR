#_______ Library Import _______ #
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from datetime import datetime
import base64
from PIL import Image
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import markdown_functions as md


#_______ Page Setup _______ #
st.set_page_config(
    page_title="NASA Weather Exploration",
    page_icon="./images/mars_icon.jpg",
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
    st.title("Mars Weather")

with col_2:
    st.write("")
    st.image("images/logo_cs.png", width=200)

with col_3:
    st.write("")


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
merged_df['temperature_2m_smooth'] = merged_df['temperature_2m'].rolling(window=7).mean()

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=("Mars Min Temp", "Mars Max Temp", "Mars Pressure", "Earth Temperature"))

# Plot each variable on a separate subplot
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['min_temp'], name='Mars Min Temp'), row=1, col=1)
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['max_temp'], name='Mars Max Temp'), row=2, col=1)
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['pressure'], name='Mars Pressure'), row=3, col=1)
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['temperature_2m_smooth'], name='Earth Temperature Smoothed'), row=4, col=1)

fig.update_xaxes(title_text="Date", row=4, col=1, rangeslider_visible=True, rangeselector=dict(
    buttons=list([
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=2, label="2y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(step="all")
    ])
))

# Update x-axis and y-axis labels
fig.update_xaxes(title_text="Date", row=4, col=1)  # Only the last subplot needs the x-axis label
fig.update_yaxes(title_text="Degrees C", row=1, col=1)
fig.update_yaxes(title_text="Degrees C", row=2, col=1)
fig.update_yaxes(title_text="Pressure (Pa)", row=3, col=1)
fig.update_yaxes(title_text="Degrees C", row=4, col=1)

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  font=dict(color='black'),
                  width=1500, 
                  height=1000, 
                  title_text="Time Series of Mars and Earth Weather Data")
st.plotly_chart(fig, theme=None)