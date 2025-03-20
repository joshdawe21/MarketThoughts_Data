import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title for the Streamlit app
st.title("Data Visualization for Dollar Index (Past 10 Years)")

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

# Plotting the data
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('WhiteSmoke')

colors = ['black', 'blue', 'green', 'red', 'purple', 'orange']
ax1.tick_params(axis='y', labelcolor='black')

for i, lookup in enumerate(lookup_columns):
    # Plot each lookup column and show its latest value in the legend label
    ax1.plot(data_filtered['Date'], data_filtered[lookup],
             label=f"{lookup}: {data_filtered[lookup].iloc[-1]:.2f}",
             color=colors[i % len(colors)])

fig.tight_layout()
ax1.legend(loc='best')
plt.grid(True)

# Display the plot in the Streamlit app
st.pyplot(fig)
