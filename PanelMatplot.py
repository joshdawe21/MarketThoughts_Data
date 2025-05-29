import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import panel as pn
from datetime import datetime
from scipy.stats import linregress
pn.extension('matplotlib')

# Set your data URL
DATA_URL = "https://raw.githubusercontent.com/joshdawe21/MarketThoughts_Data/refs/heads/main/Data.csv"

plt.rcParams['font.family'] = 'Times New Roman'   # Change to any installed font family
plt.rcParams['font.size'] = 15           # Change default font size

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
    fig = plt.figure(figsize=(15, 10))
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
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax1.set_title(
        f"Technical Analysis: {lookup}",
        fontweight='bold',
        loc='left'  # aligns title to the left
        )
    ax1.set_xticks([])  # Avoid warning: use set_xticks instead of set_xticklabels([])

    # Subplot 2: RSI
    ax2.plot(data['Date'], data['14d_RSI'], label=f"14d RSI: {latest_data_point['14d_RSI']:.2f}", color='purple')
    ax2.axhline(70, linestyle='--', color='red')
    ax2.axhline(30, linestyle='--', color='green')
    ax2.legend(loc="upper left")
    ax2.grid(True)
    ax2.set_xticks([])

    # Subplot 3: Std Dev
    ax3.plot(data['Date'], data['20d_std'], label=f"20d St.Dev: {latest_data_point['20d_std']:.2f}")
    ax3.legend(loc="upper left")
    ax3.grid(True)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=0.2)
    plt.close(fig)

    return pn.pane.Matplotlib(fig, tight=True)

def seasonality(lookup: str = 'Dollar Index', yearlookup: int = 5, data_url: str = DATA_URL) -> pn.pane.Matplotlib:
    data = pd.read_csv(data_url)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data[lookup] = pd.to_numeric(data[lookup], errors='coerce')

    if yearlookup != 0:
        cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=yearlookup)
        data = data[data['Date'] >= cutoff_date]

    data['Year'] = data['Date'].dt.year
    data['Week'] = data['Date'].dt.isocalendar().week

    weekly_avg = data.groupby(['Year', 'Week'])[lookup].mean().reset_index()
    weekly_stats = data.groupby('Week')[lookup].agg(['min', 'max', 'mean']).reset_index()
    weekly_stats.columns = ['Week', 'Min', 'Max', 'Avg']
    weekly_stats['Smoothed_Avg'] = weekly_stats['Avg'].rolling(window=2, center=True).mean()

    week_to_month = {
        1: 'Jan', 5: 'Feb', 9: 'Mar', 13: 'Apr', 18: 'May', 22: 'Jun',
        27: 'Jul', 31: 'Aug', 36: 'Sep', 40: 'Oct', 45: 'Nov', 49: 'Dec'
    }
    month_positions = list(week_to_month.keys())
    month_labels = list(week_to_month.values())

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('WhiteSmoke')
    current_year = datetime.now().year

    ax.fill_between(weekly_stats['Week'], weekly_stats['Min'], weekly_stats['Max'], color='gray', alpha=0.2, label="Min-Max")

    for year in sorted(data['Year'].unique()):
        yearly_weekly = weekly_avg[weekly_avg['Year'] == year]
        color = 'black' if year == current_year else None
        ax.plot(yearly_weekly['Week'], yearly_weekly[lookup], label=str(year), color=color)

    ax.plot(weekly_stats['Week'], weekly_stats['Smoothed_Avg'], label='Weekly Avg', linestyle='-', color='blue')
    ax.set_title(
        f"Seasonality: {lookup}",
        fontweight='bold',
        loc='left'  # aligns title to the left
        )
    ax.set_xticks(range(1, 54, 4))
    ax.set_xlabel('Weeks')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(month_positions)
    ax2.set_xticklabels(month_labels)
    ax2.set_xlabel("Months")

    ax.legend(loc="upper left")
    ax.grid(True)
    
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=0.2)
    plt.close(fig)
    return pn.pane.Matplotlib(fig, tight=True)


def weekly_return_distribution(lookup: str = 'Dollar Index', yearlookup: int = 5, data_url: str = DATA_URL) -> pn.pane.Matplotlib:
    from scipy.stats import norm

    data = pd.read_csv(data_url)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data[lookup] = pd.to_numeric(data[lookup], errors='coerce')

    if yearlookup != 0:
        cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=yearlookup)
        data = data[data['Date'] >= cutoff_date]

    data.set_index('Date', inplace=True)
    weekly_returns = data[lookup].resample('W').last().pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='WhiteSmoke')

    def plot_best_fit(ax, returns, bins=30):
        mu, sigma = norm.fit(returns)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        scaled_p = p * len(returns) * (xmax - xmin) / bins
        ax.plot(x, scaled_p, 'r--', linewidth=2, label=f'Fit: μ={mu:.2f}, σ={sigma:.2f}')

    ax.hist(weekly_returns, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plot_best_fit(ax, weekly_returns)

    recent_weekly_return = weekly_returns.iloc[-1]
    recent_weekly_date = weekly_returns.index[-1]
    ax.axvline(recent_weekly_return, color='red', linestyle='dashed', linewidth=2,
               label=f'Recent Weekly: {recent_weekly_return:.2f}% ({recent_weekly_date.strftime("%b %d, %Y")})')

    ax.set_title(
        f"Weekly Return Distribution: {lookup}",
        fontweight='bold',
        loc='left'  # aligns title to the left
        )
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=0.2)
    plt.close(fig)
    return pn.pane.Matplotlib(fig, tight=True)


def historical_separate_axis(lookup_columns: str, yearlookup: int = 5, data_url: str = DATA_URL) -> pn.pane.Matplotlib:
    import matplotlib.pyplot as plt
    import pandas as pd
    
    data = pd.read_csv(data_url)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    lookup_list = [col.strip() for col in lookup_columns.split(",")]

    for col in lookup_list:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    if yearlookup != 0:
        cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=yearlookup)
        data = data[data['Date'] >= cutoff_date]

    last_values = {col: data[col].iloc[-1] if not data[col].empty else None for col in lookup_list}

    fig, ax1 = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('WhiteSmoke')

    color_sequence = ['black', 'blue', 'green', 'red', 'purple', 'orange']
    ax1.plot(
        data['Date'],
        data[lookup_list[0]],
        color=color_sequence[0],
        label=f"{lookup_list[0]} ({last_values[lookup_list[0]]:.2f})"
    )
    ax1.set_title(
        f"Historical: {lookup_list}",
        fontweight='bold',
        loc='left'  # aligns title to the left
        )    
    ax1.tick_params(axis='y', labelcolor=color_sequence[0])

    ax_list = [ax1]

    for i in range(1, len(lookup_list)):
        ax_new = ax1.twinx()
        ax_list.append(ax_new)
        color = color_sequence[i % len(color_sequence)]
        ax_new.spines['right'].set_position(('outward', 20 * i))
        ax_new.plot(
            data['Date'],
            data[lookup_list[i]],
            color=color,
            label=f"{lookup_list[i]} ({last_values[lookup_list[i]]:.2f})"
        )
        ax_new.tick_params(axis='y', labelcolor=color)

    # Combine legends from all axes
    handles, labels = [], []
    for ax in ax_list:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    ax1.legend(handles, labels, loc="upper left")
    ax1.set_xlabel("Date")
    ax1.grid(True)
    
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=0.2)
    plt.close(fig)
    return pn.pane.Matplotlib(fig, tight=True)


def linear_regression(lookup_columns, yearlookup: int = 5, data_url: str = DATA_URL) -> pn.pane.Matplotlib:
    data = pd.read_csv(data_url)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    lookup_list = [col.strip() for col in lookup_columns.split(",")]

    if len(lookup_list) < 2:
        raise ValueError("Please provide at least two columns for regression analysis.")

    for col in lookup_list[:2]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    if yearlookup != 0:
        cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=yearlookup)
        data = data[data['Date'] >= cutoff_date]

    data['Year'] = data['Date'].dt.year

    x_col, y_col = lookup_list[:2]
    most_recent_row = data.iloc[-1]
    most_recent_x = most_recent_row[x_col]
    most_recent_y = most_recent_row[y_col]

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('WhiteSmoke')

    unique_years = sorted(data['Year'].unique())
    colors = plt.cm.tab10(range(len(unique_years)))

    for i, year in enumerate(unique_years):
        year_data = data[data['Year'] == year]
        x = year_data[x_col]
        y = year_data[y_col]
        slope, intercept, r_value, _, _ = linregress(x, y)
        ax.scatter(x, y, label=f"{year} (R²={r_value**2:.2f})", color=colors[i], alpha=0.8)
        ax.plot(x, slope * x + intercept, color=colors[i], linestyle='--', linewidth=1)

    ax.scatter(most_recent_x, most_recent_y, color='red', s=100, label='Most Recent', zorder=5)
    ax.annotate(f"({most_recent_x:.2f}, {most_recent_y:.2f})",
                xy=(most_recent_x, most_recent_y),
                xytext=(most_recent_x + 0.1, most_recent_y + 0.1),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=10, color='red')

    slope_all, intercept_all, r_val_all, _, _ = linregress(data[x_col], data[y_col])
    ax.plot(data[x_col], slope_all * data[x_col] + intercept_all,
            color='black', linewidth=2, label=f"Overall Fit (R²={r_val_all**2:.2f})")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(
        f"Linear Regression: {lookup_list}",
        fontweight='bold',
        loc='left'  # aligns title to the left
        )
    ax.legend(title="Year",loc="upper left")
    ax.grid(True)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=0.2)
    plt.close(fig)
    return pn.pane.Matplotlib(fig, tight=True)



# --- Dashboard Setup ---
def create_dashboard(yearlookup: int):
    fig1 = technical_analysis('Dollar Index', yearlookup)
    fig2 = seasonality('Dollar Index', yearlookup)
    fig3 = weekly_return_distribution('Dollar Index', yearlookup)
    fig4 = historical_separate_axis('Dollar Index, US02Y Yield (%), US10Y Yield (%)', yearlookup)
    fig5 = linear_regression('US02Y Yield (%), Dollar Index', yearlookup)
    fig6 = linear_regression('US10Y Yield (%), Dollar Index', yearlookup)

    dashboard = pn.GridSpec(sizing_mode='scale_width', max_width=1200)

    dashboard[0, 0] = fig1
    dashboard[0, 1] = fig2
    dashboard[1, 0] = fig3
    dashboard[1, 1] = fig4
    dashboard[2, 0] = fig5
    dashboard[2, 1] = fig6
   
    dashboard.spacing = (10, 10)  # Horizontal, vertical spacing
    dashboard.margin = (5, 5, 5, 5)

    return dashboard


year_selector = pn.widgets.Select(
    name='Select Timeframe',
    options={'1Y': 1, '5Y': 5, "10Y": 10, 'All': 0},
    value=10
)

@pn.depends(year_selector)
def update_dashboard(yearlookup):
    return create_dashboard(yearlookup)

title = pn.pane.Markdown(
    "<div style='font-size:24px; font-weight:bold; font-family:\"Times New Roman\";'>"
    "Dollar Index Dashboard"
    "</div>",
    sizing_mode='stretch_width'
)

sources = pn.pane.Markdown(
    "<div style='font-size:15px;font-family:\"Times New Roman\";'>"
    "Sources: Trading View"
    "</div>",
    sizing_mode='stretch_width'
)

pn.Column(
    title,
    year_selector,
    pn.layout.Spacer(height=1),  # Small vertical gap
    update_dashboard,
    sources,
    margin=0
).servable(title="Dollar Index Dashboard with Rate Comparison and Regression")
