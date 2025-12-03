from dash import dcc, html
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import date, timedelta


def prediction(stock, n_days):
    try:
        df = yf.download(stock, period="6mo")

        if df.empty:
            raise ValueError("No data available for forecasting.")

        df = df.reset_index()
        df["Day"] = np.arange(len(df))

        X = df[["Day"]].values
        y = df["Close"].values

        # Simple 90/10 split to avoid empty x_test
        split = int(len(X) * 0.9)
        x_train, x_test = X[:split], X[split:]
        y_train = y[:split]

        if len(x_train) < 5:
            raise ValueError("Not enough data to train model.")

        # Simple SVR (no GridSearch — stable)
        svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        svr.fit(x_train, y_train)

        last_day = X[-1][0]
        future_days = np.array([[last_day + i] for i in range(1, n_days + 1)])

        future_dates = [
            date.today() + timedelta(days=i)
            for i in range(1, n_days + 1)
        ]

        preds = svr.predict(future_days)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=preds,
                mode="lines+markers",
                name="Forecast"
            )
        )

        fig.update_layout(
            title=f"{stock.upper()} — Next {n_days} Days Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Close Price"
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Forecast Error: {str(e)}",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
