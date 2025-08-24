import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('vehicles_us.csv')

# Page config
st.set_page_config(page_title="Car Sales Dashboard", layout="wide")

# App title
st.title("🚗 U.S. Car Sales Dashboard")

st.markdown("Explore car sales listings with interactive filters and visualizations. Use the sidebar to narrow results by vehicle type, condition, and price range.")

# Sidebar filters
st.sidebar.header("🔍 Filter Listings")

# Type filter
car_types = df['type'].dropna().unique()
selected_types = st.sidebar.multiselect("Select car types", sorted(car_types), default=car_types)

# Condition filter
conditions = df['condition'].dropna().unique()
selected_conditions = st.sidebar.multiselect("Select conditions", sorted(conditions), default=conditions)

# Price range slider
min_price, max_price = st.sidebar.slider(
    "Select price range ($)",
    int(df['price'].min()), int(df['price'].max()),
    (int(df['price'].min()), int(df['price'].max()))
)

# Filter dataset
filtered_df = df[
    (df['type'].isin(selected_types)) &
    (df['condition'].isin(selected_conditions)) &
    (df['price'] >= min_price) &
    (df['price'] <= max_price)
]

# Summary stats
st.subheader("📊 Current Selection Overview")
st.write(f"Showing **{filtered_df.shape[0]:,}** listings")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("💲 Price Distribution")
    fig_price = px.histogram(filtered_df, x='price', nbins=50)
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("📍 Odometer vs Price")
    fig_scatter = px.scatter(filtered_df, x='odometer', y='price', color='condition')
    st.plotly_chart(fig_scatter, use_container_width=True)

# Optional checkbox for more
if st.checkbox("📦 Show Price by Vehicle Condition"):
    st.subheader("🛠️ Price by Condition")
    fig_box = px.box(filtered_df, x='condition', y='price', color='condition')
    st.plotly_chart(fig_box, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Made with ❤️ by miatho25 · [GitHub Repo](https://github.com/miatho25/SDT-project4)")

