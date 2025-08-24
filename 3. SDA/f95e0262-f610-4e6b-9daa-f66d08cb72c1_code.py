# In[5]
# Loading all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[8]
# Load the data files into different DataFrames
users = pd.read_csv('/datasets/megaline_users.csv')
calls = pd.read_csv('/datasets/megaline_calls.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')
plans = pd.read_csv('/datasets/megaline_plans.csv')

# In[12]
# Print the general/summary information about the plans' DataFrame
print(users.info())
print(calls.info())
print(messages.info())
print(internet.info())
print(plans.info())

# In[13]
#Print a sample of data for plans

plans.head

# In[19]
# Convert 'mb_per_month_included' from MB to GB #removing mb column
plans['gb_per_month_included'] = plans['mb_per_month_included'] / 1024
plans.drop(columns=['mb_per_month_included'], inplace=True)  

# Check and enforce correct data types
numeric_columns = [
    'messages_included', 'minutes_included', 'usd_monthly_pay',
    'usd_per_gb', 'usd_per_message', 'usd_per_minute', 'gb_per_month_included'
]
plans[numeric_columns] = plans[numeric_columns].astype(float)  # making sure they are floats

# Check for missing values across datasets
print(users.isnull().sum())  
print(calls.isnull().sum())  
print(messages.isnull().sum())  
print(internet.isnull().sum())  
print(plans.isnull().sum())  

# Handle missing churn_date by filling with 'Not Churned'
users['churn_date'].fillna('Not Churned', inplace=True)

# Verify corrections
print(plans.head())
print(users.info())

# In[23]

# Calculate cost per included unit
plans['cost_per_included_gb'] = plans['usd_monthly_pay'] / plans['gb_per_month_included']
plans['cost_per_included_minute'] = plans['usd_monthly_pay'] / plans['minutes_included']
plans['cost_per_included_message'] = plans['usd_monthly_pay'] / plans['messages_included']

# Calculate overage cost multiplier
plans['overage_cost_multiplier_gb'] = plans['usd_per_gb'] / plans['cost_per_included_gb']
plans['overage_cost_multiplier_minute'] = plans['usd_per_minute'] / plans['cost_per_included_minute']
plans['overage_cost_multiplier_message'] = plans['usd_per_message'] / plans['cost_per_included_message']

# Compute a value score (higher score = better value)
plans['value_score'] = (plans['minutes_included'] + plans['messages_included'] + plans['gb_per_month_included']) / plans['usd_monthly_pay']

# Verify the enriched dataset
print(plans.head())

# In[26]
# Print the general/summary information about the users' DataFrame

print(users.describe())

# In[27]
# Print a sample of data for users

print(users.head())

# In[32]
#convert reg_date and churn_date to datetime and change not churned to Nan
users['reg_date'] = pd.to_datetime(users['reg_date'])
users['churn_date'] = users['churn_date'].replace("Not Churned", np.nan)
users['churn_date'] = pd.to_datetime(users['churn_date'])

#format city
users['city'] = users['city'].str.replace(" MSA", "", regex=False)

# Extract state abbreviation (last two characters after comma)
users['state'] = users['city'].str.extract(r', (\w\w)$')


print(users.head())

#check for duplicates
print(users.duplicated().sum()) 

#check for missing values
print(users.isnull().sum())

# In[36]
# Calculate User Tenure (Time Since Registration)
#Adding a tenure_months column to see how long each user has been active
today = pd.to_datetime('2018-12-31') 
users['tenure_months'] = ((today - users['reg_date']) / np.timedelta64(1, 'M')).astype(int)  
#This will give the number of months each user has been subscribed.

#Categorize Users by Age Group
bins = [0, 18, 30, 45, 60, np.inf]
labels = ['Teen', 'Young Adult', 'Middle-aged', 'Older Adult', 'Senior']
users['age_group'] = pd.cut(users['age'], bins=bins, labels=labels)
#This can useful for analysis

#Add a "Churned" Column (Yes/No)
#Instead of checking NaT, create a column to indicate whether a user has churned
users['churned'] = users['churn_date'].notna().astype(int)  # 1 = Churned, 0 = Active

print(users.head())

# In[38]
# Print the general/summary information about the calls' DataFrame

print(calls.describe())

# In[39]
# Print a sample of data for calls
print(calls.head())

# In[44]
#convert call_date to datetime format
calls['call_date'] = pd.to_datetime(calls['call_date'])

#round up the calls
calls['duration'] = np.ceil(calls['duration'])

#check for missing or zero length calls
calls = calls[calls['duration'] > 0]

print(calls.head())

# In[47]
# Add a "Day of the Week" column:
calls['day_of_week'] = calls['call_date'].dt.day_name()

# Categorize Calls by Time of Day:
calls['call_hour'] = calls['call_date'].dt.hour
calls['time_of_day'] = pd.cut(
    calls['call_hour'], bins=[0, 6, 12, 18, 24], 
    labels=['Night', 'Morning', 'Afternoon', 'Evening'], include_lowest=True
)
print(calls.head())

# In[49]
# Print the general/summary information about the messages' DataFrame

print(messages.describe())

# In[50]
# Print a sample of data for messages
print(messages.head())

# In[55]
# convert message_date to datetime format
messages['message_date'] = pd.to_datetime(messages['message_date'])

# Check for missing values
print(messages.isnull().sum())

# Check for duplicate messages
print(f"Duplicate rows: {messages.duplicated().sum()}")

# Display updated sample
print(messages.head())

# In[58]
#  Additional factors to enhance the the data would be:

# Ensure message_date is in datetime format
messages['message_date'] = pd.to_datetime(messages['message_date'])

# Weekday vs. Weekend Messaging, some users may send more messages on weekdays versus weekends.
messages['day_of_week'] = messages['message_date'].dt.day_name()
messages['is_weekend'] = messages['day_of_week'].isin(['Saturday', 'Sunday'])

# Monthly Message Count per User,helps analyze user engagement trends over time. 
messages['year_month'] = messages['message_date'].dt.to_period('M')
monthly_messages = messages.groupby(['user_id', 'year_month']).size().reset_index(name='message_count')

# Ensure session_date is in datetime format
internet['session_date'] = pd.to_datetime(internet['session_date'])

# Total Data Usage per User per Month
internet['year_month'] = internet['session_date'].dt.to_period('M')
monthly_data_usage = internet.groupby(['user_id', 'year_month'])['mb_used'].sum().reset_index(name='total_mb_used')

# convert mb to gb and round up at the end of month
internet['gb_used'] = np.ceil(internet['mb_used'] / 1024)

print(messages.info())

# In[61]
# Print the general/summary information about the internet DataFrame
print(internet.describe())

# In[62]
# Print a sample of data for the internet traffic

print(internet.head())

# In[67]
# If year_month is redundant, remove it and re-calculate
internet = internet.drop(columns=['year_month'], errors='ignore')

# Convert session_date to datetime format
internet['session_date'] = pd.to_datetime(internet['session_date'])

# Check for missing values
missing_values = internet.isnull().sum()
print("Missing Values:\n", missing_values)

# Print sample data to confirm changes
print(internet.head())

# In[70]
# Additional factors to enhance data

# Extract day of the week from session_date
internet['day_of_week'] = internet['session_date'].dt.day_name()

# Flag for weekends (Saturday and Sunday)
internet['is_weekend'] = internet['day_of_week'].isin(['Saturday', 'Sunday'])

# Flag sessions where data usage is zero
internet['is_zero_usage'] = internet['mb_used'] == 0

print(internet.head())

# In[73]
# Print out the plan conditions and make sure they are clear for you

print(plans)

# In[75]
# Calculate the number of calls made by each user per month. Save the result.
# Convert session_date and message_date to datetime for proper aggregation
calls['call_date'] = pd.to_datetime(calls['call_date'])
messages['message_date'] = pd.to_datetime(messages['message_date'])
internet['session_date'] = pd.to_datetime(internet['session_date'])

# Aggregate the number of calls per user per month
calls['year_month'] = calls['call_date'].dt.to_period('M')
calls_per_user = calls.groupby(['user_id', 'year_month']).size().reset_index(name='calls_count')

print("Calls Per User Per Month:\n", calls_per_user)

# In[76]
# Calculate the amount of minutes spent by each user per month. Save the result.
# the total duration(minutes) of calls per user per month

call_duration_per_user = calls.groupby(['user_id', 'year_month'])['duration'].sum().reset_index(name='total_call_duration')
print("\nCall Duration Per User Per Month:\n", call_duration_per_user)

# In[77]
# Calculate the number of messages sent by each user per month. Save the result.

# Aggregate the number of messages sent per user per month
messages['year_month'] = messages['message_date'].dt.to_period('M')
messages_per_user = messages.groupby(['user_id', 'year_month']).size().reset_index(name='messages_count')
print("\nMessages Per User Per Month:\n", messages_per_user)

# In[78]
# Calculate the volume of internet traffic used by each user per month. Save the result.

# Aggregate the total internet traffic used per user per month
internet['year_month'] = internet['session_date'].dt.to_period('M')
internet_traffic_per_user = internet.groupby(['user_id', 'year_month'])['mb_used'].sum().reset_index(name='total_mb_used')

# Convert the total internet traffic from MB to GB for ease of analysis
internet_traffic_per_user['total_gb_used'] = internet_traffic_per_user['total_mb_used'] / 1024

print("\nInternet Traffic Per User Per Month:\n", internet_traffic_per_user)

# In[80]
# Merge the data for calls, minutes, messages, internet based on user_id and month

aggregated_data = calls_per_user.merge(call_duration_per_user, on=['user_id', 'year_month'], how='left')
aggregated_data = aggregated_data.merge(messages_per_user, on=['user_id', 'year_month'], how='left')
aggregated_data = aggregated_data.merge(internet_traffic_per_user, on=['user_id', 'year_month'], how='left')

# Fill missing values with appropriate defaults
aggregated_data.fillna({'calls_count': 0, 'total_call_duration': 0, 'messages_count': 0, 'total_mb_used': 0}, inplace=True)

# Merge with users data to include plan and state information
users['state'] = users['city'].str.extract(r', (\w\w)$')  # Extract state from city
aggregated_data = aggregated_data.merge(users[['user_id', 'plan', 'state']], on='user_id', how='left')

# Fill any remaining missing values
aggregated_data['plan'].fillna('Unknown', inplace=True)
aggregated_data['state'].fillna('Unknown', inplace=True)

# Print the final dataset
print(aggregated_data.head())

# In[81]
# Add the plan information

aggregated_data = aggregated_data.merge(users[['user_id', 'plan']], on='user_id', how='left')

# If there are duplicate 'plan' columns, merge them into one
aggregated_data['plan'] = aggregated_data[['plan_x', 'plan_y']].bfill(axis=1).iloc[:, 0]

# Drop the extra columns
aggregated_data = aggregated_data.drop(columns=['plan_x', 'plan_y'], errors='ignore')


print(aggregated_data)

# In[83]
# Calculate the monthly revenue for each user

# Define the plan conditions
plan_conditions = {
    'ultimate': {
        'free_minutes': 1000,
        'free_messages': 500,
        'free_data_gb': 10,
        'monthly_fee': 70,  # Fixed monthly fee
        'call_rate': 0.10,  # Cost per extra minute
        'message_rate': 0.05,  # Cost per extra message
        'data_rate': 10  # Cost per extra GB
    },
    'surf': {
        'free_minutes': 500,
        'free_messages': 200,
        'free_data_gb': 5,
        'monthly_fee': 30,  # Fixed monthly fee
        'call_rate': 0.15,  # Cost per extra minute
        'message_rate': 0.10,  # Cost per extra message
        'data_rate': 15  # Cost per extra GB
    }
}


# Function to calculate the revenue for a given row (user per month)# Function to calculate the revenue for a given row (user per month)
def calculate_revenue(row):
    plan = str(row['plan']).lower().strip()  # Normalize the plan name
    
    if plan not in plan_conditions:
        print(f"Warning: Unknown plan '{plan}' for user {row['user_id']}")  # Debugging output
        return 0  # If no valid plan is found, return 0 revenue
    
    plan_data = plan_conditions[plan]  # Get plan details
    
    # Get actual usage, defaulting to 0 if the column is missing
    used_minutes = row.get('total_call_duration', 0)
    used_messages = row.get('messages_count', 0)
    used_data_gb = row.get('total_mb_used', 0) / 1024  # Convert MB to GB

    # Calculate overages (extra usage beyond the free limit)
    extra_minutes = max(0, used_minutes - plan_data['free_minutes'])
    extra_messages = max(0, used_messages - plan_data['free_messages'])
    extra_data = max(0, used_data_gb - plan_data['free_data_gb'])

    # Compute extra costs
    extra_call_cost = extra_minutes * plan_data['call_rate']
    extra_message_cost = extra_messages * plan_data['message_rate']
    extra_data_cost = extra_data * plan_data['data_rate']

    # Total revenue: Fixed monthly fee + extra charges
    total_revenue = plan_data['monthly_fee'] + extra_call_cost + extra_message_cost + extra_data_cost
    return round(total_revenue, 2)  # Round for better readability

# Apply the revenue calculation function
aggregated_data['monthly_revenue'] = aggregated_data.apply(calculate_revenue, axis=1)
    
print(aggregated_data[['user_id', 'year_month', 'plan', 'monthly_revenue']].head())

# In[86]
# Compare average duration of calls per each plan per each distinct month. Plot a bar plat to visualize it.
# Calculate average call duration per plan per month
avg_call_duration = aggregated_data.groupby(['year_month', 'plan'])['total_call_duration'].mean().reset_index()

# Rename column for clarity
avg_call_duration.rename(columns={'total_call_duration': 'avg_call_duration'}, inplace=True)

# Display the first few rows
print(avg_call_duration.head())

# In[87]
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_call_duration, x='year_month', y='avg_call_duration', hue='plan', palette='coolwarm')

# Beautify the plot
plt.title('Average Call Duration Per Plan Per Month')
plt.xlabel('Month')
plt.ylabel('Average Call Duration (minutes)')
plt.xticks(rotation=45)  # Rotate month labels for better readability
plt.legend(title='Plan')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# In[88]

# Compare the number of minutes users of each plan require each month. Plot a histogram.

# Group the data by 'plan' and 'year_month' and calculate the total call duration per user
monthly_call_duration = aggregated_data.groupby(['plan', 'year_month'])['total_call_duration'].sum().reset_index()

# Set the plot size
plt.figure(figsize=(12, 6))

# Plot histogram for each plan's total call duration per month
for plan in monthly_call_duration['plan'].unique():
    plan_data = monthly_call_duration[monthly_call_duration['plan'] == plan]
    plt.hist(plan_data['total_call_duration'], bins=20, alpha=0.6, label=plan)

# Customize the plot
plt.title('Histogram of Total Call Duration per Plan per Month')
plt.xlabel('Total Call Duration (Minutes)')
plt.ylabel('Frequency')
plt.legend(title='Plan')

# Show the plot
plt.tight_layout()
plt.show()

# In[90]
# Calculate the mean and the variance of the monthly call duration

# Group the data by 'plan' and 'year_month' and calculate the total call duration per user
monthly_call_duration = aggregated_data.groupby(['plan', 'year_month'])['total_call_duration'].sum().reset_index()

# Calculate the mean and variance of the total call duration for each plan per month
call_stats = monthly_call_duration.groupby('plan')['total_call_duration'].agg(['mean', 'var']).reset_index()

# Print the results
print(call_stats)

# In[91]
# Plot a boxplot to visualize the distribution of the monthly call duration
# Prepare data for plotting
plans = aggregated_data['plan'].unique()
plan_data = [aggregated_data[aggregated_data['plan'] == plan]['total_call_duration'] for plan in plans]

# Create a boxplot to visualize the distribution of the monthly call duration per plan
plt.figure(figsize=(12, 6))
plt.boxplot(plan_data, labels=plans)

# Customize the plot
plt.title('Distribution of Monthly Call Duration per Plan')
plt.xlabel('Plan')
plt.ylabel('Total Call Duration (minutes)')

# Display the plot
plt.show()

# In[95]
# Compare the number of messages users of each plan tend to send each month

# Step 1: Create a new 'year_month' column based on message_date
messages['year_month'] = messages['message_date'].dt.to_period('M')

# Step 2: Group by 'user_id' and 'year_month', then count the messages for each user per month
monthly_messages = messages.groupby(['user_id', 'year_month']).size().reset_index(name='total_messages_sent')

# Step 3: Merge with the aggregated data to include the plan information
monthly_messages_with_plan = pd.merge(monthly_messages, aggregated_data[['user_id', 'plan']], on='user_id', how='left')

# Step 4: Group by 'plan' and 'year_month' to get the total number of messages sent per plan
monthly_messages_per_plan = monthly_messages_with_plan.groupby(['plan', 'year_month'])['total_messages_sent'].sum().reset_index()

# Step 5: Set the plot size for better readability
plt.figure(figsize=(12, 6))

# Step 6: Plot a histogram for each plan's total number of messages sent per month
for plan in monthly_messages_per_plan['plan'].unique():
    plan_data = monthly_messages_per_plan[monthly_messages_per_plan['plan'] == plan]
    plt.hist(plan_data['total_messages_sent'], bins=20, alpha=0.6, label=plan)

# Step 7: Customize the plot
plt.title('Histogram of Total Messages Sent per Plan per Month')
plt.xlabel('Total Messages Sent')
plt.ylabel('Frequency')
plt.legend(title='Plan')

# Step 8: Show the plot
plt.tight_layout()
plt.show()

# In[96]
# Compare the amount of internet traffic consumed by users per plan

# Step 1: Create a new 'year_month' column from 'session_date'
internet['year_month'] = internet['session_date'].dt.to_period('M')

# Step 2: Group by 'user_id' and 'year_month' and sum total internet usage per user per month
monthly_internet_usage = internet.groupby(['user_id', 'year_month'])['mb_used'].sum().reset_index()

# Step 3: Merge with the aggregated data to get the plan information
monthly_internet_with_plan = pd.merge(monthly_internet_usage, aggregated_data[['user_id', 'plan']], on='user_id', how='left')

# Step 4: Group by 'plan' and 'year_month' to get total internet usage per plan per month
monthly_internet_per_plan = monthly_internet_with_plan.groupby(['plan', 'year_month'])['mb_used'].sum().reset_index()

# Step 5: Set plot size
plt.figure(figsize=(12, 6))

# Step 6: Plot histogram for each plan's internet usage per month
for plan in monthly_internet_per_plan['plan'].unique():
    plan_data = monthly_internet_per_plan[monthly_internet_per_plan['plan'] == plan]
    plt.hist(plan_data['mb_used'], bins=20, alpha=0.6, label=plan)

# Step 7: Customize the plot
plt.title('Histogram of Internet Usage per Plan per Month')
plt.xlabel('Total Internet Usage (MB)')
plt.ylabel('Frequency')
plt.legend(title='Plan')

# Step 8: Show the plot
plt.tight_layout()
plt.show()

# In[108]
# Step 1: Calculate mean and variance of revenue per plan
revenue_stats = aggregated_data.groupby('plan')['monthly_revenue'].agg(['mean', 'var', 'std']).reset_index()
print(revenue_stats)

# Step 2: Boxplot to visualize revenue distribution per plan
plt.figure(figsize=(10, 6))
plans = aggregated_data['plan'].unique()
plan_data = [aggregated_data[aggregated_data['plan'] == plan]['monthly_revenue'] for plan in plans]
plt.boxplot(plan_data, labels=plans)

plt.title('Revenue Distribution per Plan')
plt.xlabel('Plan')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()

# Step 3: Histogram to visualize revenue distribution
plt.figure(figsize=(10, 6))
for plan in aggregated_data['plan'].unique():
    plan_data = aggregated_data[aggregated_data['plan'] == plan]['monthly_revenue']
    plt.hist(plan_data, bins=20, alpha=0.6, label=plan)

plt.title('Histogram of Revenue per Plan')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.legend(title='Plan')
plt.grid(True)
plt.show()

# In[117]
# *Formulating the Hypotheses*
# H₀ (Null Hypothesis): There is no significant difference in average revenue between the two plans.
# H₁ (Alternative Hypothesis): The average revenue differs significantly between the two plans.



# Test the hypotheses

import scipy.stats as stats

# Extract revenue data for each plan
surf_revenue = aggregated_data[aggregated_data['plan'] == 'surf']['monthly_revenue']
ultimate_revenue = aggregated_data[aggregated_data['plan'] == 'ultimate']['monthly_revenue']

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(surf_revenue, ultimate_revenue, equal_var=False)

# Print results
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# Conclusion based on alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The average revenue differs significantly between the two plans.")
else:
    print("Fail to reject the null hypothesis: No significant difference in average revenue between the two plans.")

# In[120]
# Test the hypotheses

#from scipy import stats

# Define NY-NJ area users
ny_nj_users = aggregated_data[aggregated_data['state'].isin(['NY', 'NJ'])]

# Define users from other regions
other_region_users = aggregated_data[~aggregated_data['state'].isin(['NY', 'NJ'])]

# Extract revenue data
ny_nj_revenue = ny_nj_users['monthly_revenue']
other_region_revenue = other_region_users['monthly_revenue']

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(ny_nj_revenue, other_region_revenue, equal_var=False)

# Print results
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# Conclusion based on alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The average revenue differs significantly between NY-NJ users and other regions.")
else:
    print("Fail to reject the null hypothesis: No significant difference in average revenue between NY-NJ users and other regions.")

