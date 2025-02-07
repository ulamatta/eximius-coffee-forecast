import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from prophet import Prophet

# Read and preprocess the CSV data, parsing dates in the "ds" column.
df = pd.read_csv("sales.csv", parse_dates=["ds"])

# Drop the item id column if it exists.
if "item id" in df.columns:
    df.drop("item id", axis=1, inplace=True)

# Get a list of unique coffee types from the "ingredient" column.
coffee_types = df["ingredient"].unique()

def run_forecast(coffee, forecast_period=12):
    """
    For the selected coffee type, aggregate monthly sales and build a Prophet model.
    
    Parameters:
        coffee (str): The selected coffee type (e.g., "Colombia")
        forecast_period (int): Number of months to forecast into the future.
    
    Returns:
        forecast (DataFrame): The Prophet forecast dataframe (including future dates)
        df_coffee (DataFrame): The monthly aggregated actual sales data
    """
    # Filter data for the selected coffee type.
    df_coffee = df[df["ingredient"] == coffee].copy()
    
    # Aggregate sales on a monthly basis using Month End frequency ("ME").
    df_coffee = df_coffee.set_index("ds").resample("ME").sum().reset_index()
    
    # Initialize and fit the Prophet model.
    # Since the data is monthly, only yearly seasonality is enabled.
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_coffee)
    
    # Create future dates with monthly frequency ("ME") and generate the forecast.
    future = model.make_future_dataframe(periods=forecast_period, freq='ME')
    forecast = model.predict(future)
    return forecast, df_coffee

# Initialize the Dash app using the Darkly theme.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.title = "Coffee Forecast Dashboard"

# Define the app layout.
app.layout = html.Div(
    [
        dbc.Container(
            [
                # Header
                dbc.Row(
                    dbc.Col(
                        html.H1(
                            "Coffee Forecast Dashboard",
                            className="text-center text-white mb-4",
                        ),
                        width=12,
                    )
                ),
                # Dropdown for selecting coffee type
                dbc.Row(
                    dbc.Col(
                        dcc.Dropdown(
                            id="coffee-dropdown",
                            options=[{"label": coffee, "value": coffee} for coffee in coffee_types],
                            value=coffee_types[0],
                            clearable=False,
                            style={"color": "#000"},
                        ),
                        width=4,
                    ),
                    className="mb-4",
                ),
                # Slider for adjusting forecast horizon (in months)
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.Label("Forecast Horizon (Months)", style={"color": "white"}),
                                dcc.Slider(
                                    id="forecast-horizon-slider",
                                    min=1,
                                    max=24,
                                    step=1,
                                    value=12,
                                    marks={1: '1', 6: '6', 12: '12', 18: '18', 24: '24'},
                                ),
                            ]
                        ),
                        width=6,
                    ),
                    className="mb-4",
                ),
                # KPI Cards Row
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Total Sales (lbs)", className="card-title"),
                                        html.H2(id="total-sales", className="card-text"),
                                    ]
                                ),
                                color="primary",
                                inverse=True,
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Avg Monthly Sales (lbs)", className="card-title"),
                                        html.H2(id="avg-sales", className="card-text"),
                                    ]
                                ),
                                color="secondary",
                                inverse=True,
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Next Month Forecast (lbs)", className="card-title"),
                                        html.H2(id="next-day-forecast", className="card-text"),
                                    ]
                                ),
                                color="info",
                                inverse=True,
                            ),
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                # Forecast Graph Row
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(id="forecast-graph"),
                        width=12,
                    )
                ),
            ],
            fluid=True,
            className="mt-4",
        )
    ],
    style={"backgroundColor": "#2c3e50", "minHeight": "100vh", "padding": "10px"},
)

# Callback to update the forecast and KPI cards based on the selected coffee type and forecast horizon.
@app.callback(
    [
        Output("forecast-graph", "figure"),
        Output("total-sales", "children"),
        Output("avg-sales", "children"),
        Output("next-day-forecast", "children"),
    ],
    [
        Input("coffee-dropdown", "value"),
        Input("forecast-horizon-slider", "value")
    ]
)
def update_forecast(coffee, forecast_period):
    # Run forecast for the selected coffee type using monthly aggregated data.
    forecast, df_coffee = run_forecast(coffee, forecast_period)
    
    # Calculate KPI metrics from the monthly aggregated historical data.
    total_sales = df_coffee["y"].sum()
    avg_sales = df_coffee["y"].mean()

    # Determine the next month (using MonthEnd to align with the aggregated data).
    last_date = df_coffee["ds"].max()
    next_month = last_date + pd.offsets.MonthEnd(1)

    # Retrieve the forecasted value for the next month; if not available, use "N/A".
    next_month_data = forecast[forecast["ds"] == next_month]
    if not next_month_data.empty:
        next_month_forecast_value = round(next_month_data["yhat"].iloc[0], 2)
    else:
        next_month_forecast_value = "N/A"

    # Create a Plotly figure for the historical monthly data and forecast.
    fig = go.Figure()

    # Plot the actual (historical) monthly aggregated data.
    fig.add_trace(
        go.Scatter(
            x=df_coffee["ds"],
            y=df_coffee["y"],
            mode="markers+lines",
            name="Actual",
            marker=dict(color="#1f77b4"),
        )
    )

    # Plot the forecasted values.
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="#ff7f0e"),
        )
    )

    # Plot the forecast uncertainty intervals as a shaded area.
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.2)",
            line=dict(width=0),
            showlegend=True,
            name="Confidence Interval",
            hoverinfo="skip",
        )
    )

    # Update the layout for a dark theme.
    fig.update_layout(
        template="plotly_dark",
        title=f"Monthly Forecast for {coffee}",
        xaxis_title="Date",
        yaxis_title="Sales (lbs)",
        hovermode="x",
    )

    # Format KPI numbers with commas as thousand separators.
    total_sales_str = f"{total_sales:,.2f}"
    avg_sales_str = f"{avg_sales:,.2f}"
    if isinstance(next_month_forecast_value, (int, float)):
        next_month_forecast_str = f"{next_month_forecast_value:,.2f}"
    else:
        next_month_forecast_str = next_month_forecast_value

    return (
        fig,
        total_sales_str,
        avg_sales_str,
        next_month_forecast_str,
    )

if __name__ == "__main__":
    app.run_server(debug=True)
