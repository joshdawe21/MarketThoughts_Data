import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import panel as pn

pn.extension()

# Set your data URL
DATA_URL = "https://raw.githubusercontent.com/joshdawe21/MarketThoughts_Data/refs/heads/main/Data.csv"

# --- Function 1: Technical Analysis ---
def technical_analysis(lookup: str, yearlookup: int, data_url: str = DATA_URL) -> pn.pane.Matplotlib:
    data = pd.read_csv(data_url)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data[lookup] = pd.to_numeric(data[lookup], errors='coerce')
    
    if yearlookup != 0:
        cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=yearlookup)
        data = data[data['Date'] >= cutoff_date]

    # First Subplot calculations
    data['20d_MA'] = data[lookup].rolling(window=20).mean()
    data['50d_MA'] = data[lookup].rolling(window=50).mean()
    data['100d_MA'] = data[lookup].rolling(window=100).mean()
    data['BB_upper'] = data['20d_MA'] + 2 * data[lookup].rolling(window=20).std()
    data['BB_lower'] = data['20d_MA'] - 2 * data[lookup].rolling(window=20).std()

    # Second Subplot calculations
    delta = data[lookup].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['14d_RSI'] = 100 - (100 / (1 + rs))

    # Third Subplot calculations
    data['20d_std'] = data[lookup].rolling(window=20).std()

    # Get the latest data point for annotation
    latest_data_point = data.iloc[-1]

    # Plotting
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('WhiteSmoke')
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 0.25, 0.25])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Subplot 1: MAs and BBs
    ax1.plot(data['Date'], data[lookup], label=f"{lookup}: {latest_data_point[lookup]:.2f}", color='black')
    ax1.plot(data['Date'], data['20d_MA'], label=f"20d MA: {latest_data_point['20d_MA']:.2f}", color='green')
    ax1.plot(data['Date'], data['50d_MA'], label=f"50d MA: {latest_data_point['50d_MA']:.2f}", color='orange')
    ax1.plot(data['Date'], data['100d_MA'], label=f"100d MA: {latest_data_point['100d_MA']:.2f}", color='red')
    ax1.plot(data['Date'], data['BB_upper'], linestyle='--', color='blue')
    ax1.plot(data['Date'], data['BB_lower'], label=f"2 St.Dev BB: Upper {latest_data_point['BB_upper']:.2f}, Lower {latest_data_point['BB_lower']:.2f}", linestyle='--', color='blue')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticklabels([])

    # Subplot 2: RSI
    ax2.plot(data['Date'], data['14d_RSI'], label=f"14d RSI: {latest_data_point['14d_RSI']:.2f}", color='purple')
    ax2.axhline(70, linestyle='--', color='red')
    ax2.axhline(30, linestyle='--', color='green')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticklabels([])

    # Subplot 3: Std Dev
    ax3.plot(data['Date'], data['20d_std'], label=f"20d St.Dev: {latest_data_point['20d_std']:.2f}")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    return pn.pane.Matplotlib(fig, tight=True)

# Placeholder stubs for the other functions
def seasonality(*args, **kwargs):
    return pn.pane.Markdown("Seasonality chart placeholder")

def weekly_return_distribution(*args, **kwargs):
    return pn.pane.Markdown("Weekly Return Distribution placeholder")

def historical_separate_axis(*args, **kwargs):
    return pn.pane.Markdown("Historical Separate Axis placeholder")

def linear_regression(*args, **kwargs):
    return pn.pane.Markdown("Linear Regression placeholder")

# --- Dashboard Setup ---
def create_dashboard(yearlookup: int):
    fig1 = technical_analysis('Dollar Index', yearlookup)
    fig2 = seasonality()
    fig3 = weekly_return_distribution()
    fig4 = historical_separate_axis()
    fig5 = linear_regression()

    dashboard = pn.GridSpec(sizing_mode='stretch_width', max_width=1200)
    dashboard[0:2, 0] = fig1
    dashboard[0, 1] = fig2
    dashboard[1, 1] = fig3
    dashboard[2, 0] = fig4
    dashboard[2, 1] = fig5

    return dashboard

# Dropdown for year lookup
year_selector = pn.widgets.Select(name='Select Timeframe', options={'1 Year': 1, '5 Years': 5, 'All Data': 0}, value=5)

@pn.depends(year_selector)
def update_dashboard(yearlookup):
    return create_dashboard(yearlookup)

pn.Column(year_selector, update_dashboard).servable(title="Dollar Index Dashboard with Rate Comparison and Regression")
