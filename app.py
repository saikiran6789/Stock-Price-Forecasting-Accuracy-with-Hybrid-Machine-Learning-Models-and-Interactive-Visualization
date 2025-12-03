import dash
from dash import dcc
from dash import html
from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
# model
from model import prediction
from sklearn.svm import SVR

import requests

def get_company_profile(ticker):
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        profile = data["quoteSummary"]["result"][0]["assetProfile"]

        summary = profile.get("longBusinessSummary", "No business summary available.")
        industry = profile.get("industry", "N/A")
        sector = profile.get("sector", "N/A")

        return summary, industry, sector

    except Exception:
        return "No business summary available.", "N/A", "N/A"

def get_stock_price_fig(df):
    fig = px.line(df,
                  x="Date",
                  y=["Close", "Open"],
                  title="Closing and Opening Price vs Date")
    return fig


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
    ])
server = app.server
# html layout of site
app.layout = html.Div(
    [
        html.Div(
            [
                # Navigation
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div([
                    html.P("Input stock code: "),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text"),
                        html.Button("Submit", id='submit'),
                    ],
                             className="form")
                ],
                         className="input-place"),
                html.Div([
                    dcc.DatePickerRange(id='my-date-picker-range',
                                        min_date_allowed=dt(1995, 8, 5),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date()),
                ],
                         className="date"),
                html.Div([
                    html.Button(
                        "Stock Price", className="stock-btn", id="stock"),
                    html.Button("Indicators",
                                className="indicators-btn",
                                id="indicators"),
                    dcc.Input(id="n_days",
                              type="text",
                              placeholder="number of days"),
                    html.Button(
                        "Forecast", className="forecast-btn", id="forecast")
                ],
                         className="buttons"),
                # here
            ],
            className="nav"),

        # content
        html.Div(
            [
                html.Div(
                    [  # header
                        html.Img(id="logo"),
                        html.P(id="ticker")
                    ],
                    className="header"),
                html.Div(id="description", className="decription_ticker"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content")
            ],
            className="content"),
    ],
    className="container")


# callback for company info
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("ticker", "children"),
        Output("stock", "n_clicks"),
        Output("indicators", "n_clicks"),
        Output("forecast", "n_clicks")
    ],
    [Input("submit", "n_clicks")],
    [State("dropdown_tickers", "value")]
)
def update_data(n, val):

    if n is None:
        return (
            "Hey there! Please enter a legitimate stock code to get details.",
            "assets/logo.png",
            "Stocks",
            None, None, None
        )

    if not val:
        return (
            "Please enter a stock symbol.",
            "assets/logo.png",
            "Invalid",
            None, None, None
        )

    ticker = yf.Ticker(val)

    # Safe fallback values
    logo = "https://via.placeholder.com/150"
    name = val.upper()
    summary = "No business summary available."

    try:
        # Fast info is SAFE (rarely rate-limited)
        fast = ticker.fast_info

        name = fast.get("shortName", name)
        logo = fast.get("logo_url", logo)

        # Summary is often rate-limited, wrap in try
        try:
            info = ticker.get_info()
            summary = info.get("longBusinessSummary", summary)
        except:
            summary = "Business summary unavailable due to rate limiting."

    except:
        pass  # keep fallback values

    return summary, logo, name, None, None, None

# callback for stocks graphs
@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def stock_price(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    # Flatten MultiIndex columns
    df.columns = [col[0] for col in df.columns]

    # Reset the index to make 'Date' a column
    df.reset_index(inplace=True)

    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]


# callback for indicators
@app.callback([Output("main-content", "children")], [
    Input("indicators", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    # Flatten MultiIndex columns
    df_more.columns = [col[0] for col in df_more.columns]

    # Reset the index to make 'Date' a column
    df_more.reset_index(inplace=True)

    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


# callback for forecast
@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    
    # Check if n_days is None or not a valid integer
    if n_days is None or not n_days.isdigit():
        return [html.Div("Please enter a valid number of days for forecasting.", style={"color": "red"})]
    
    # Convert n_days to integer
    n_days = int(n_days)
    
    # Ensure n_days is at least 1
    if n_days < 1:
        return [html.Div("Number of days must be at least 1.", style={"color": "red"})]
    
    # Call the prediction function
    fig = prediction(val, n_days + 1)
    return [dcc.Graph(figure=fig)]


if __name__ == '__main__':
    app.run(debug=True)