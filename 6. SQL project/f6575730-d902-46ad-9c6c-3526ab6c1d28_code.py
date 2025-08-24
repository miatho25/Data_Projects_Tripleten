# In[2]
# Import all required libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# In[3]
# Set the style for our plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# In[5]
# Import the first dataset - Taxi companies and number of rides
taxi_companies = pd.read_csv('/datasets/project_sql_result_01.csv')
print("Taxi Companies Dataset:")
print(taxi_companies.head())
print("\nData types:")
print(taxi_companies.dtypes)
print("\nDataset shape:", taxi_companies.shape)

# In[6]
# Import the second dataset - Neighborhoods and average trips
neighborhoods = pd.read_csv('/datasets/project_sql_result_04.csv')
print("\nNeighborhoods Dataset:")
print(neighborhoods.head())
print("\nData types:")
print(neighborhoods.dtypes)
print("\nDataset shape:", neighborhoods.shape)

# In[7]
# Identify top 10 neighborhoods by dropoffs
top_10_neighborhoods = neighborhoods.sort_values(by='average_trips', ascending=False).head(10)
print("\nTop 10 neighborhoods by average number of dropoffs:")
print(top_10_neighborhoods)

# In[8]
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

# Plot 1: Taxi companies and number of rides
sns.barplot(x='trips_amount', y='company_name', data=taxi_companies, ax=ax1)
ax1.set_title('Number of Rides by Taxi Company (Nov 15-16, 2017)', fontsize=16)
ax1.set_xlabel('Number of Rides', fontsize=14)
ax1.set_ylabel('Taxi Company', fontsize=14)

# Plot 2: Top 10 neighborhoods by number of dropoffs
sns.barplot(x='average_trips', y='dropoff_location_name', data=top_10_neighborhoods, ax=ax2)
ax2.set_title('Top 10 Neighborhoods by Average Number of Dropoffs (Nov 2017)', fontsize=16)
ax2.set_xlabel('Average Number of Dropoffs', fontsize=14)
ax2.set_ylabel('Neighborhood', fontsize=14)

plt.tight_layout()
plt.show()

# In[9]
# Calculate total rides across all companies
total_rides = taxi_companies['trips_amount'].sum()

# Calculate market share for each company
taxi_companies['market_share'] = (taxi_companies['trips_amount'] / total_rides) * 100

# Sort by market share in descending order
taxi_companies_sorted = taxi_companies.sort_values(by='market_share', ascending=False)

# Display the market share data
print("\nTaxi Company Market Share (Nov 15-16, 2017):")
print(taxi_companies_sorted[['company_name', 'trips_amount', 'market_share']])

# Calculate market concentration metrics
top_3_share = taxi_companies_sorted.iloc[0:3]['market_share'].sum()
top_5_share = taxi_companies_sorted.iloc[0:5]['market_share'].sum()
herfindahl_index = sum((taxi_companies['market_share']/100)**2)

print(f"\nMarket Concentration Metrics:")
print(f"Top 3 companies market share: {top_3_share:.2f}%")
print(f"Top 5 companies market share: {top_5_share:.2f}%")
print(f"Herfindahl-Hirschman Index: {herfindahl_index:.4f} (0.25+ indicates high concentration)")

# In[10]
# Create pie chart for visual representation
plt.figure(figsize=(12, 8))

# For readability, group smaller companies into "Other" category
threshold = 5.0  # Companies with less than 5% market share will be grouped
major_companies = taxi_companies_sorted[taxi_companies_sorted['market_share'] >= threshold].copy()
other_companies = taxi_companies_sorted[taxi_companies_sorted['market_share'] < threshold].copy()

if not other_companies.empty:
    other_total = other_companies['market_share'].sum()
    other_row = pd.DataFrame({
        'company_name': ['Other'],
        'trips_amount': [other_companies['trips_amount'].sum()],
        'market_share': [other_total]
    })
    plot_data = pd.concat([major_companies, other_row], ignore_index=True)
else:
    plot_data = major_companies

# Create the pie chart
plt.pie(plot_data['market_share'], labels=plot_data['company_name'], 
        autopct='%1.1f%%', startangle=90, shadow=True, explode=[0.05]*len(plot_data))
plt.axis('equal')
plt.title('Taxi Companies Market Share in Chicago (Nov 15-16, 2017)', fontsize=16)
plt.tight_layout()
plt.show()

# In[16]
# Import the dataset for hypothesis testing
loop_ohare_trips = pd.read_csv('/datasets/project_sql_result_07.csv')

print("\nLoop to O'Hare Trips Dataset:")
print(loop_ohare_trips.head())
print("\nData types:")
print(loop_ohare_trips.dtypes)
print("\nDataset shape:", loop_ohare_trips.shape)

# In[17]
# Convert start_ts to datetime
loop_ohare_trips['start_ts'] = pd.to_datetime(loop_ohare_trips['start_ts'])

# Basic statistics of the dataset
print("\nSummary statistics for the Loop to O'Hare trips:")
print(loop_ohare_trips.describe())

# In[18]
# Check for missing values
print("\nMissing values in Loop to O'Hare dataset:")
print(loop_ohare_trips.isnull().sum())

# In[19]
# Count of trips by weather condition
weather_counts = loop_ohare_trips['weather_conditions'].value_counts()
print("\nCount of trips by weather condition:")
print(weather_counts)

# In[20]
# Calculate average duration for each weather condition
avg_duration_by_weather = loop_ohare_trips.groupby('weather_conditions')['duration_seconds'].mean()
print("\nAverage duration by weather condition (in seconds):")
print(avg_duration_by_weather)

# In[21]
# Visualize the distribution of trip durations by weather conditions
plt.figure(figsize=(10, 6))
sns.boxplot(x='weather_conditions', y='duration_seconds', data=loop_ohare_trips)
plt.title('Trip Duration Distribution by Weather Condition', fontsize=16)
plt.xlabel('Weather Condition', fontsize=14)
plt.ylabel('Duration (seconds)', fontsize=14)
plt.show()

# In[27]
# Hypothesis Testing
# Separate the data by weather condition
good_weather_trips = loop_ohare_trips[loop_ohare_trips['weather_conditions'] == 'Good']['duration_seconds']
bad_weather_trips = loop_ohare_trips[loop_ohare_trips['weather_conditions'] == 'Bad']['duration_seconds']

# Set significance level
alpha = 0.05
print(f"\nSignificance level (alpha): {alpha}")

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(bad_weather_trips, good_weather_trips, equal_var=False)
print(f"\nT-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# In[28]
# Interpret the results
print("\nHypothesis testing conclusion:")
if p_value < alpha:
    print(f"We reject the null hypothesis (p-value = {p_value:.4f} < {alpha}).")
    print("There is a statistically significant difference in the average ride duration between good and bad weather conditions.")
else:
    print(f"We fail to reject the null hypothesis (p-value = {p_value:.4f} > {alpha}).")
    print("There is no statistically significant difference in the average ride duration between good and bad weather conditions.")

# In[29]
# Effect size (Cohen's d)
mean_diff = bad_weather_trips.mean() - good_weather_trips.mean()
pooled_std = np.sqrt((bad_weather_trips.var() * (len(bad_weather_trips) - 1) + 
                     good_weather_trips.var() * (len(good_weather_trips) - 1)) / 
                     (len(bad_weather_trips) + len(good_weather_trips) - 2))
cohens_d = abs(mean_diff) / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
if cohens_d < 0.2:
    effect_interpretation = "small"
elif cohens_d < 0.5:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"This represents a {effect_interpretation} effect size.")

print(f"\nThe difference in average ride duration: {abs(mean_diff):.2f} seconds")
print(f"Average duration in good weather: {good_weather_trips.mean():.2f} seconds")
print(f"Average duration in bad weather: {bad_weather_trips.mean():.2f} seconds")

