import pandas as pd
import plotly.graph_objects as go

# Load the data from the URL
url = "https://raw.githubusercontent.com/joshdawe21/MarketThoughts_Data/349f25b1f000ac4e09f18122d61eb589c2d9659f/Data.csv"
data = pd.read_csv(url)

# Define the lookup parameters
lookup_columns = ["Dollar Index"]  # Only lookup "Dollar Index"
yearlookup = 10  # Use data from the past 10 years

# Ensure the 'Date' column is in datetime format (interpreting dates as day/month/year)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Convert the lookup column(s) to numeric (coerce any errors to NaN)
for lookup in lookup_columns:
    data[lookup] = pd.to_numeric(data[lookup], errors='coerce')

# Filter the data for the past 'yearlookup' years
cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=yearlookup)
data_filtered = data[data['Date'] >= cutoff_date]

# Define colors to use for each trace
colors = ['black', 'blue', 'green', 'red', 'purple', 'orange']

# Create a Plotly figure
fig = go.Figure()

# Add a trace for each lookup column
for i, lookup in enumerate(lookup_columns):
    fig.add_trace(go.Scatter(
        x=data_filtered['Date'], 
        y=data_filtered[lookup],
        mode='lines',
        name=f"{lookup}: {data_filtered[lookup].iloc[-1]:.2f}",
        line=dict(color=colors[i % len(colors)])
    ))

# Add layout details including gridlines and background color
fig.update_layout(
    title="Market Data Over the Past 10 Years",
    xaxis_title="Date",
    yaxis_title="Value",
    plot_bgcolor="whitesmoke",
    legend=dict(x=0, y=1),
    margin=dict(b=100)  # Extra bottom margin for the annotation
)

# Show grid lines on both axes
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Create an annotation with a clickable URL (data source)
annotation_text = f'<a href="{url}" target="_blank">View Data Source</a>'
fig.update_layout(
    annotations=[dict(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=annotation_text,
        font=dict(size=14)
    )]
)

# Save the figure as an HTML file and open it in the default web browser
fig.write_html("market_data_plot.html", auto_open=True)
