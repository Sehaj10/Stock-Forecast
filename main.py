import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast')
selected_stock  = st.text_input('Enter Stock Ticker')

n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

#@st.cache_data
def load_data(ticker):
    for ticker in selected_stock:
        company = yf.Ticker(selected_stock).info
        company_name = company['longName']
        st.subheader(company_name)
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
        continue

        


data = load_data(selected_stock)

if data is not None:
    st.write('Raw data')
    st.write(data.tail())

# Plot raw data
def plot_raw_data():
    if data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.update_layout(title_text= 'Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.write("Please type a valid ticker symbol")

plot_raw_data()

if data is not None:
# Predict forecast with Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
else:
        st.write("")
