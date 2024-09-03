#_______ Library Import _______ #
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from PIL import Image
import markdown_functions as md
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats


#_______ Page Setup _______ #
st.set_page_config(
    page_title="NASA Weather Exploration",
    page_icon = "🌎",
    layout = "wide",
    initial_sidebar_state = "collapsed"
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

st.markdown(md.get_navbar_markdown(), unsafe_allow_html = True)
st.markdown(md.background_and_tabs_styles(encoded_image), unsafe_allow_html = True)
st.markdown(md.get_global_theme_styles(), unsafe_allow_html = True)
st.markdown(md.get_selectbox_styles(), unsafe_allow_html = True)


# _______ Page _______ #
locations = ["Amazon", "Athens", "Valley", "Everest", "London", "McMurdo", "Oymyakon", "Sahara", "Serengeti", "Sydney", "Zurich", ]

st.write("")
st.title("")
col_1, col_2, col_3 = st.columns([8, 2, 0.5])

with col_1:  
    st.title("Weather Visualization")

with col_2:
    st.write("")
    st.image("images/logo_cs.png", width = 200)

with col_3:
    st.write("")



# loads and caches data for the selected location
@st.cache_data
def load_data(location):
    return pd.read_parquet(f"./data/{location}.parquet.gzip")


location = st.selectbox("Select Location", options = locations)

data = load_data(location)
features = ["Temperature", "Rain", "Humidity", "Direct Radiation"]


feature_columns = {
    "Temperature": "temperature_2m",
    "Rain": "rain",
    "Humidity": "relative_humidity_2m",
    "Radiation": "direct_radiation_instant"
}
selected_feature = st.selectbox("Select Feature to Plot", options = features)

selected_column_name = feature_columns[selected_feature]

fig = px.line(data, x = "date", y = selected_column_name, title = f"{selected_feature} Over Time in {location}",
                 width = 1300, height = 500)
fig.update_layout(paper_bgcolor = "rgba(242, 245, 250, 0.4)", 
                    plot_bgcolor = "rgba(242, 245, 250, 0.2)",

                    font=dict(size = 12,
                              color = "black"))
fig.update_traces(line_color = "#7792E3")
st.plotly_chart(fig, theme = None)


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

st.subheader("Mars Minimum Temperature vs. Earth Temperature (Solar Flare Impact by Intensity)")

fig2 = px.scatter(merged_df, x="min_temp", y="temperature_2m_smooth", color="intensity",
                  title="Mars Minimum Temperature vs. Earth Temperature (Solar Flare Impact by Intensity)",
                  labels={"min_temp": "Mars Min Temp", "temperature_2m_smooth": "Earth Temperature"},
                  category_orders={"intensity": ["Low", "Medium", "High"]})  # Assuming 'Low', 'Medium', 'High' are the intensity levels
fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  font=dict(color='black'),
                  height=500, 
                  width=1300)
st.plotly_chart(fig2, theme=None)

#__________________________________________________________

st.subheader("Solar Flare Impact")

# Solar Flare Impact Visualization:
fig3 = px.line(merged_df, x="date", y=["temperature_2m_smooth", "solar_flare"],
              title="Earth Temperature and Solar Flare Events",
              labels={"date": "Date", "temperature_2m": "Earth Temperature", "solar_flare": "Solar Flare"})

# include a range slider and range selector
fig3.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='black'),
    height=1000,
    width=2000
)
st.plotly_chart(fig3, theme=None)
