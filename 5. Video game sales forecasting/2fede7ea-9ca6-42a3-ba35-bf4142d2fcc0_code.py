# In[2]
# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)

# In[4]
# Load the dataset
df = pd.read_csv('/datasets/games.csv')

# Display first few rows
df.head()

# In[5]
# Display basic information about the dataset
df.info()

# In[6]
# Check for duplicate entries
df.duplicated().sum()

# In[9]
print(f"Total number of records: {df.shape[0]}")

# In[12]
# Check missing values
df.isnull().sum()

# In[15]
# Check unique values in some key columns
print("Unique platforms:", df['Platform'].unique())
print("Unique genres:", df['Genre'].unique())
print("Unique ratings:", df['Rating'].unique())

# In[18]
# Convert column names to lowercase
df.columns = df.columns.str.lower()

# In[19]
# Verify the changes
df.columns

# In[21]
# Check current data types
df.dtypes

# In[22]
# Make changes to data types if necessary

# Convert 'year_of_release' to integer (optional: can leave as float if NaN exists)

# Coerce any problematic values to NaN
df['year_of_release'] = pd.to_numeric(df['year_of_release'], errors='coerce').astype('Int64')

# In[23]
# Pay attention to the abbreviation TBD (to be determined). Specify how you intend to handle such cases.

# Ensure 'user_score' is treated as string before converting
df['user_score'] = df['user_score'].astype(str)

# Convert 'user_score' to numeric, coercing 'TBD' and other non-numeric values to NaN
df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

#check current data types
df.dtypes

# In[25]
# Examine missing values

# Count missing values in each column
missing_values = df.isnull().sum()

# Display only columns with missing values
missing_values[missing_values > 0]

# In[26]
# Calculate percentage of missing values

# Percentage of missing values
missing_percent = (df.isnull().sum() / len(df)) * 100

# Display only columns with missing data
missing_percent[missing_percent > 0].sort_values(ascending=False)

# In[27]
# Calculate missing values and percentage as a DataFrame
missing_values = df.isnull().sum().to_frame(name='Missing Values')
missing_values['% Missing'] = round((df.isnull().sum() / len(df)) * 100, 2)

# Sort by percentage of missing values (descending)
missing_values = missing_values[missing_values['Missing Values'] > 0].sort_values(by='% Missing', ascending=False)

# Display the result
missing_values

# In[30]
# Handle missing values based on analysis
# Your code here to handle missing values according to your strategy

# 1. Drop rows where 'year_of_release' or 'genre' is missing
df = df.dropna(subset=['year_of_release', 'genre'])

# 2. Fill missing values in 'rating' with 'Unknown'
df['rating'] = df['rating'].fillna('Unknown')

# 3. Ensure 'user_score' is numeric (convert 'TBD' to NaN already done earlier)
# If not done already, make sure:
df['user_score'] = df['user_score'].astype(str)
df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

# 4. Leave 'user_score' and 'critic_score' as-is (NaN will be ignored in correlations/visualizations)

# 5. Optionally drop rows where 'name' is missing 
df = df.dropna(subset=['name'])

# Final check on missing values
df.isnull().sum()

# In[32]
# Calculate total sales across all regions and put them in a different column

# Calculate total sales by summing across all regions
df['total_sales'] = df[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

# Preview the updated DataFrame
df[['name', 'na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales']].head()

# In[34]
# Create a DataFrame with game releases by year
games_per_year = df['year_of_release'].value_counts().sort_index()

# In[35]
# Visualize the distribution of games across years
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
games_per_year.plot(kind='bar')
plt.title('Number of Games Released per Year')
plt.xlabel('Year of Release')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# In[36]
# Display summary statistics for each year
games_per_year.describe()

# In[39]
# Calculate total sales by platform and year
platform_year_sales = df.groupby(['year_of_release', 'platform'])['total_sales'].sum().unstack().fillna(0)

# Preview the table
platform_year_sales.tail()

# In[40]
# Create a heatmap of platform sales over time
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))
sns.heatmap(platform_year_sales.T, cmap='YlGnBu', linewidths=0.5)

plt.title('Platform Sales by Year (in millions)', fontsize=16)
plt.xlabel('Year of Release')
plt.ylabel('Platform')
plt.tight_layout()
plt.show()

# In[41]
# Identify platforms with declining sales

# Total sales by platform and year
platform_trends = df.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

# Pivot the table for easier plotting
pivot_table = platform_trends.pivot(index='year_of_release', columns='platform', values='total_sales')

# Focus on platforms with the highest overall sales
top_platforms = df.groupby('platform')['total_sales'].sum().sort_values(ascending=False).head(10).index

# Plot trends for those platforms
plt.figure(figsize=(14, 8))

for platform in top_platforms:
    plt.plot(pivot_table.index, pivot_table[platform], label=platform)

plt.title('Total Sales Over Time by Platform')
plt.xlabel('Year of Release')
plt.ylabel('Total Sales (millions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# In[44]
# Your code here to filter the dataset to relevant years
# Example:
# relevant_years = [XXXX, XXXX, XXXX] # Replace with your chosen years
# df_relevant = df[df['year_of_release'].isin(relevant_years)]

# Filter to relevant years (2012–2016)
relevant_years = list(range(2012, 2017))
df_relevant = df[df['year_of_release'].isin(relevant_years)]

# Confirm the filter
df_relevant['year_of_release'].value_counts().sort_index()


# Justify your choice with data

# Game count per year
games_per_year = df.groupby('year_of_release')['name'].count()

# Total global sales per year
sales_per_year = df.groupby('year_of_release')['total_sales'].sum()

# Combine into a single DataFrame
yearly_summary = pd.DataFrame({
    'Number of Games': games_per_year,
    'Total Global Sales (millions)': sales_per_year
})

# Filter for the last 10 years (2007–2016) to show the trend
yearly_summary_recent = yearly_summary.loc[2007:2016]

# Display the summary
display(yearly_summary_recent)

# In[45]
#updated time period selection

# Filter to relevant years (2014–2016) based on recent market trends
relevant_years = list(range(2014, 2017))
df_relevant = df[df['year_of_release'].isin(relevant_years)]

# Confirm the filter
df_relevant['year_of_release'].value_counts().sort_index()

# Justification

# Game count per year
games_per_year = df.groupby('year_of_release')['name'].count()

# Total global sales per year
sales_per_year = df.groupby('year_of_release')['total_sales'].sum()

# Combine into a single DataFrame
yearly_summary = pd.DataFrame({
    'Number of Games': games_per_year,
    'Total Global Sales (millions)': sales_per_year
})

# Focus on recent years only
yearly_summary_recent = yearly_summary.loc[2013:2016]

# Display the summary
display(yearly_summary_recent)

# In[49]
# Analyze platform sales trends

# Total sales by platform for the selected period (2012–2016)
platform_sales = df_relevant.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

# In[50]
# Sort platforms by total sales
# Display sorted sales
platform_sales

# In[51]
# Visualize top platforms

# Plot total sales of top platforms
plt.figure(figsize=(10, 6))
platform_sales.plot(kind='bar')
plt.title('Total Sales by Platform (2012–2016)')
plt.ylabel('Total Sales (millions)')
plt.xlabel('Platform')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Calculate year-over-year growth for each platform

# Group by year and platform, then sum total sales
platform_yearly_sales = df_relevant.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

# Pivot table: years as rows, platforms as columns
platform_sales_pivot = platform_yearly_sales.pivot(index='year_of_release', columns='platform', values='total_sales').fillna(0)

# Calculate YoY growth rates for each platform
platform_growth = platform_sales_pivot.pct_change().fillna(0) * 100  # Convert to percentage

# Display growth table
platform_growth.round(2).tail()

# Your code here to calculate and visualize platform growth rates

# Group by year and platform, then sum total sales
platform_yearly_sales = df_relevant.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

# Pivot table: years as rows, platforms as columns
platform_sales_pivot = platform_yearly_sales.pivot(index='year_of_release', columns='platform', values='total_sales').fillna(0)

# Calculate YoY growth rates for each platform
platform_growth = platform_sales_pivot.pct_change().fillna(0) * 100  # Convert to percentage

# Display growth table
platform_growth.round(2).tail()

# In[53]
# Create box plot of sales by platform

# Set figure size
plt.figure(figsize=(14, 7))

# Create a box plot of total sales per platform
sns.boxplot(data=df_relevant, x='platform', y='total_sales', showfliers=False)

# Enhance readability
plt.yscale('log')  # Log scale to handle outliers
plt.title('Distribution of Global Sales by Platform (2012–2016)')
plt.xlabel('Platform')
plt.ylabel('Global Sales (millions, log scale)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# In[55]
# Calculate detailed statistics for each platform

# Group by platform and describe sales stats
platform_stats = df_relevant.groupby('platform')['total_sales'].describe().round(2)

# Display results
platform_stats

# In[56]
# Choose a popular platform based on your previous analysis

# Filter the relevant dataset for PS4 only
ps4_data = df_relevant[df_relevant['platform'] == 'PS4']

# Preview data
ps4_data[['name', 'critic_score', 'user_score', 'total_sales']].head()

# In[57]
# Create scatter plots for both critic and user scores

# In[58]
# Critic Scores
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ps4_data, x='critic_score', y='total_sales')
plt.title('Critic Score vs. Total Sales (PS4)')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales (millions)')
plt.grid(True)
plt.tight_layout()
plt.show()

# User Scores
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ps4_data, x='user_score', y='total_sales')
plt.title('User Score vs. Total Sales (PS4)')
plt.xlabel('User Score')
plt.ylabel('Total Sales (millions)')
plt.grid(True)
plt.tight_layout()
plt.show()


# Calculate correlations

# Correlation between critic_score and total_sales
critic_corr = ps4_data[['critic_score', 'total_sales']].corr().iloc[0, 1]

# Correlation between user_score and total_sales
user_corr = ps4_data[['user_score', 'total_sales']].corr().iloc[0, 1]

# Display results
print(f"Correlation between Critic Score and Sales (PS4): {critic_corr:.2f}")
print(f"Correlation between User Score and Sales (PS4): {user_corr:.2f}")

# In[60]
# Find games released on multiple platforms
#I will group by game name and count how many platforms each game appears on:

# Count how many platforms each game appears on
multi_platform_games = df_relevant.groupby('name')['platform'].nunique()

# Filter for games released on 2 or more platforms
multi_platform_games = multi_platform_games[multi_platform_games > 1]

# Get only the records for these multi-platform games
df_multi_platform = df_relevant[df_relevant['name'].isin(multi_platform_games.index)]

# In[61]
# Compare sales across platforms for these games
# Your code here to analyze and visualize cross-platform performance

# I'll visualize how the same game performs differently by platform and use a box plot to show general trends:

plt.figure(figsize=(14, 7))
sns.boxplot(data=df_multi_platform, x='platform', y='total_sales')

plt.title('Sales Distribution of Multi-Platform Games (2012–2016)')
plt.xlabel('Platform')
plt.ylabel('Total Sales (millions)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Compare Average sale per platform for shared games

# Calculate average sales for each game on each platform
avg_sales = df_multi_platform.groupby(['name', 'platform'])['total_sales'].mean().reset_index()

# Pivot to compare side by side
sales_comparison = avg_sales.pivot(index='name', columns='platform', values='total_sales')

# Show first few rows
sales_comparison.head()

# In[63]
# Analyze genre performance

# Total sales by genre
genre_sales = df_relevant.groupby('genre')['total_sales'].sum().sort_values(ascending=False)

# Display sales
genre_sales

# In[64]
# Sort genres by total sales

# Group by genre and sum total sales, then sort in descending order
genre_sales = df_relevant.groupby('genre')['total_sales'].sum().sort_values(ascending=False)

# Display result
genre_sales

# In[65]
# Visualize genre distribution

# Bar plot of total sales by genre
plt.figure(figsize=(12, 6))
genre_sales.plot(kind='bar', color='skyblue')

plt.title('Total Global Sales by Genre (2012–2016)')
plt.xlabel('Genre')
plt.ylabel('Total Sales (millions)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# In[66]
# Calculate market share for each genre

# Calculate percentage share of each genre
genre_market_share = (genre_sales / genre_sales.sum()) * 100


# Pie chart of market share
plt.figure(figsize=(9, 9))
plt.pie(genre_market_share, labels=genre_market_share.index, autopct='%1.1f%%', startangle=140)
plt.title('Genre Market Share (2012–2016)')
plt.tight_layout()
plt.show()

# In[69]
# Are there any genres showing recent growth or decline?
# To determine this, I can group by both genre and year, then sum total sales:

# Total sales by genre and year
genre_trends = df_relevant.groupby(['year_of_release', 'genre'])['total_sales'].sum().unstack().fillna(0)

# Plot example for top genres
genre_trends[['Action', 'Shooter', 'Sports']].plot(figsize=(12, 6), title='Top Genre Trends (2012–2016)')

# In[71]
#Average performance arcoss Genres
avg_sales_per_genre = df_relevant.groupby('genre')['total_sales'].mean().sort_values(ascending=False)
avg_sales_per_genre

# In[75]
# Function to analyze platform performance by region

def top_platforms_by_region(df, region_col, top_n=5):
    # Group by platform and sum sales for the region
    region_platform_sales = df.groupby('platform')[region_col].sum().sort_values(ascending=False).head(top_n)
    
    # Display the results
    print(f"Top {top_n} platforms in {region_col.upper()}:")
    display(region_platform_sales)
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    region_platform_sales.plot(kind='bar', color='teal')
    plt.title(f'Top {top_n} Platforms in {region_col.upper()}')
    plt.xlabel('Platform')
    plt.ylabel(f'Sales in {region_col.upper()} (millions)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# In[76]
# Analyze each region

#North America
top_platforms_by_region(df_relevant, 'na_sales')

#Europe
top_platforms_by_region(df_relevant, 'eu_sales')

#Japan
top_platforms_by_region(df_relevant, 'jp_sales')

# In[78]
# Create a comparative platform analysis

# Calculate total sales per platform by region

# Sum sales by platform and region
regional_platform_sales = df_relevant.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales']].sum()

# Sort platforms by total global sales to get top platforms
top_platforms = df_relevant.groupby('platform')['total_sales'].sum().sort_values(ascending=False).head(5).index

# Filter only top platforms
regional_top_platforms = regional_platform_sales.loc[top_platforms]

# Display table
regional_top_platforms

# In[79]
# Visualize cross-regional comparison for top platforms

# Plot grouped bar chart
regional_top_platforms.plot(kind='bar', figsize=(10, 6))
plt.title('Top 5 Platforms by Region (2012–2016)')
plt.xlabel('Platform')
plt.ylabel('Sales (millions)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(title='Region')
plt.tight_layout()
plt.show()

# In[81]
# Function to analyze genre performance by region

def top_genres_by_region(df, region_col, top_n=5):
    # Group by genre and sum sales in the specified region
    genre_sales = (
        df.groupby('genre')[region_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Plotting
    plt.figure(figsize=(8, 5))
    genre_sales.plot(kind='bar', color='coral')
    plt.title(f'Top {top_n} Genres in {region_col.upper()}')
    plt.xlabel('Genre')
    plt.ylabel(f'Sales in {region_col.upper()} (millions)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Print sales for reference
    print(f"\nTop {top_n} genres in {region_col.upper()}:")
    print(genre_sales)

    
# North America
top_genres_by_region(df_relevant, 'na_sales')

#Europe
top_genres_by_region(df_relevant, 'eu_sales')

#Japan
top_genres_by_region(df_relevant, 'jp_sales')

# In[83]
# Create a comparative genre analysis

# Group by genre and sum sales in each region
regional_genre_sales = df_relevant.groupby('genre')[['na_sales', 'eu_sales', 'jp_sales']].sum()

# Sort by total sales in NA for consistent comparison
regional_genre_sales = regional_genre_sales.sort_values(by='na_sales', ascending=False)

# Display the table
regional_genre_sales

# In[84]
# visualize genre preference

# Plot grouped bar chart
regional_genre_sales.plot(kind='bar', figsize=(12, 6))
plt.title('Genre Sales Comparison by Region (2012–2016)')
plt.xlabel('Genre')
plt.ylabel('Sales (millions)')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# In[86]
# Function to analyze ESRB rating impact

def esrb_impact_by_region(df, region_col):
    # Group by ESRB rating and sum regional sales
    esrb_sales = (
        df.groupby('rating')[region_col]
        .sum()
        .sort_values(ascending=False)
    )

    # Plot
    plt.figure(figsize=(8, 5))
    esrb_sales.plot(kind='bar', color='slateblue')
    plt.title(f'ESRB Rating Impact on Sales in {region_col.upper()}')
    plt.xlabel('ESRB Rating')
    plt.ylabel(f'Sales in {region_col.upper()} (millions)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\nESRB impact in {region_col.upper()}:")
    print(esrb_sales)

# In[87]
# Analyze ESRB impact for each region

#North America
esrb_impact_by_region(df_relevant, 'na_sales')

#Europe
esrb_impact_by_region(df_relevant, 'eu_sales')

#Japan
esrb_impact_by_region(df_relevant, 'jp_sales')

# In[89]
# Set the alpha threshold
alpha = 0.05

# In[91]
from scipy import stats

# Filter and clean user scores for Xbox One and PC
xbox_scores = df_relevant[(df_relevant['platform'] == 'XOne') & (df_relevant['user_score'].notnull())]['user_score']
pc_scores = df_relevant[(df_relevant['platform'] == 'PC') & (df_relevant['user_score'].notnull())]['user_score']

# Perform independent t-test
t_stat1, p_val1 = stats.ttest_ind(xbox_scores, pc_scores, equal_var=False)  # Welch’s t-test

print(f"T-statistic: {t_stat1:.4f}, p-value: {p_val1:.4f}")

# Interpret result
if p_val1 < alpha:
    print("We reject the null hypothesis: Average user ratings for Xbox One and PC are significantly different.")
else:
    print("We fail to reject the null hypothesis: No significant difference in user ratings between Xbox One and PC.")

# In[93]
# Filter and clean user scores for Action and Sports genres
action_scores = df_relevant[(df_relevant['genre'] == 'Action') & (df_relevant['user_score'].notnull())]['user_score']
sports_scores = df_relevant[(df_relevant['genre'] == 'Sports') & (df_relevant['user_score'].notnull())]['user_score']

# Perform independent t-test
t_stat2, p_val2 = stats.ttest_ind(action_scores, sports_scores, equal_var=False)

print(f"T-statistic: {t_stat2:.4f}, p-value: {p_val2:.4f}")

# Interpret result
if p_val2 < alpha:
    print("We reject the null hypothesis: There is a significant difference in user ratings between Action and Sports genres.")
else:
    print("We fail to reject the null hypothesis: No significant difference in user ratings between Action and Sports genres.")

