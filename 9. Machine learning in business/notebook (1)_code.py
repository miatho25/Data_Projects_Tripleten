# In[3]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the datasets
region_0 = pd.read_csv('/datasets/geo_data_0.csv')
region_1 = pd.read_csv('/datasets/geo_data_1.csv')
region_2 = pd.read_csv('/datasets/geo_data_2.csv')

# Store regions in a dictionary
regions = {
    'Region 0': region_0,
    'Region 1': region_1,
    'Region 2': region_2
}

# Quick look at the data
display(region_0.head())

# In[5]
# Explore the data structure
print("Data Shape for Each Region:")
for name, df in regions.items():
    print(f"{name}: {df.shape}")

print("\nData Info for Region 0:")
print(region_0.info())

print("\nBasic Statistics for Region 0:")
print(region_0.describe())

# In[7]
print("\nMissing Values Check:")
for name, df in regions.items():
    print(f"\n{name}:")
    print(df.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows Check:")
for name, df in regions.items():
    print(f"{name}: {df.duplicated().sum()} duplicates")

# In[8]
print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Check Region 0
print("\nREGION 0:")
print("Missing values:")
print(region_0.isnull().sum())
print(f"Duplicate rows: {region_0.duplicated().sum()}")

# Check Region 1
print("\nREGION 1:")
print("Missing values:")
print(region_1.isnull().sum())
print(f"Duplicate rows: {region_1.duplicated().sum()}")

# Check Region 2
print("\nREGION 2:")
print("Missing values:")
print(region_2.isnull().sum())
print(f"Duplicate rows: {region_2.duplicated().sum()}")

# In[9]
# Analyze distributions 

print("DISTRIBUTION ANALYSIS OF FEATURES AND TARGETS")


# Create visualizations for feature distributions
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
fig.suptitle('Distribution of Features and Target Variable by Region', fontsize=14)

for i, (region_name, df) in enumerate(regions.items()):
    # Plot f0
    axes[i, 0].hist(df['f0'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[i, 0].set_title(f'{region_name}: f0')
    axes[i, 0].set_ylabel('Frequency')
    
    # Plot f1
    axes[i, 1].hist(df['f1'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[i, 1].set_title(f'{region_name}: f1')
    
    # Plot f2
    axes[i, 2].hist(df['f2'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[i, 2].set_title(f'{region_name}: f2')
    
    # Plot product (target)
    axes[i, 3].hist(df['product'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[i, 3].set_title(f'{region_name}: product (target)')

plt.tight_layout()
plt.show()

# In[10]
# Create box plots for comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
data_for_box = []
labels = []

for region_name, df in regions.items():
    data_for_box.extend([df['f0'].values, df['f1'].values, df['f2'].values, df['product'].values])
    labels.extend([f'{region_name}\nf0', f'{region_name}\nf1', f'{region_name}\nf2', f'{region_name}\nproduct'])

ax.boxplot(data_for_box, labels=labels)
ax.set_title('Feature and Target Distributions Across All Regions')
ax.set_ylabel('Values')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# In[11]
# Statistical comparison
print("\nStatistical Summary of Features and Target:")

for region_name, df in regions.items():
    print(f"\n{region_name} Statistics:")
    print(df[['f0', 'f1', 'f2', 'product']].describe().round(2))

# In[15]
# Define features and target
features = ['f0', 'f1', 'f2']
target = 'product'

# Create empty dictionaries to store results
results_region_0 = {}
results_region_1 = {}
results_region_2 = {}

print("MODEL TRAINING - REGION 0")


# Train model for Region 0
X_0 = region_0[features]
y_0 = region_0[target]

X_train_0, X_valid_0, y_train_0, y_valid_0 = train_test_split(
    X_0, y_0, test_size=0.25, random_state=42
)

print(f"Training set size: {X_train_0.shape}")
print(f"Validation set size: {X_valid_0.shape}")

# Train the model
model_0 = LinearRegression()
model_0.fit(X_train_0, y_train_0)

# Make predictions
predictions_0 = model_0.predict(X_valid_0)

# Calculate metrics
avg_predicted_0 = predictions_0.mean()
avg_actual_0 = y_valid_0.mean()
rmse_0 = np.sqrt(mean_squared_error(y_valid_0, predictions_0))
r2_score_0 = model_0.score(X_valid_0, y_valid_0)

print(f"\nModel Performance Metrics:")
print(f"  Average actual volume: {avg_actual_0:.2f} thousand barrels")
print(f"  Average predicted volume: {avg_predicted_0:.2f} thousand barrels")
print(f"  Model RMSE: {rmse_0:.2f}")
print(f"  R² Score: {r2_score_0:.4f}")

print(f"\nModel Coefficients:")
for feature, coef in zip(features, model_0.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model_0.intercept_:.4f}")

# Store results for Region 0
results_region_0 = {
    'predictions': predictions_0,
    'actual': y_valid_0.values,
    'model': model_0,
    'avg_predicted': avg_predicted_0,
    'avg_actual': avg_actual_0,
    'rmse': rmse_0,
    'r2_score': r2_score_0
}

# In[16]

print("MODEL TRAINING - REGION 1")


# Train model for Region 1
X_1 = region_1[features]
y_1 = region_1[target]

X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)

print(f"Training set size: {X_train_1.shape}")
print(f"Validation set size: {X_valid_1.shape}")

# Train the model
model_1 = LinearRegression()
model_1.fit(X_train_1, y_train_1)

# Make predictions
predictions_1 = model_1.predict(X_valid_1)

# Calculate metrics
avg_predicted_1 = predictions_1.mean()
avg_actual_1 = y_valid_1.mean()
rmse_1 = np.sqrt(mean_squared_error(y_valid_1, predictions_1))
r2_score_1 = model_1.score(X_valid_1, y_valid_1)

print(f"\nModel Performance Metrics:")
print(f"  Average actual volume: {avg_actual_1:.2f} thousand barrels")
print(f"  Average predicted volume: {avg_predicted_1:.2f} thousand barrels")
print(f"  Model RMSE: {rmse_1:.2f}")
print(f"  R² Score: {r2_score_1:.4f}")

print(f"\nModel Coefficients:")
for feature, coef in zip(features, model_1.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model_1.intercept_:.4f}")

# Store results for Region 1
results_region_1 = {
    'predictions': predictions_1,
    'actual': y_valid_1.values,
    'model': model_1,
    'avg_predicted': avg_predicted_1,
    'avg_actual': avg_actual_1,
    'rmse': rmse_1,
    'r2_score': r2_score_1
}

# In[17]

print("MODEL TRAINING - REGION 2")


# Train model for Region 2
X_2 = region_2[features]
y_2 = region_2[target]

X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(
    X_2, y_2, test_size=0.25, random_state=42
)

print(f"Training set size: {X_train_2.shape}")
print(f"Validation set size: {X_valid_2.shape}")

# Train the model
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train_2)

# Make predictions
predictions_2 = model_2.predict(X_valid_2)

# Calculate metrics
avg_predicted_2 = predictions_2.mean()
avg_actual_2 = y_valid_2.mean()
rmse_2 = np.sqrt(mean_squared_error(y_valid_2, predictions_2))
r2_score_2 = model_2.score(X_valid_2, y_valid_2)

print(f"\nModel Performance Metrics:")
print(f"  Average actual volume: {avg_actual_2:.2f} thousand barrels")
print(f"  Average predicted volume: {avg_predicted_2:.2f} thousand barrels")
print(f"  Model RMSE: {rmse_2:.2f}")
print(f"  R² Score: {r2_score_2:.4f}")

print(f"\nModel Coefficients:")
for feature, coef in zip(features, model_2.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model_2.intercept_:.4f}")

# Store results for Region 2
results_region_2 = {
    'predictions': predictions_2,
    'actual': y_valid_2.values,
    'model': model_2,
    'avg_predicted': avg_predicted_2,
    'avg_actual': avg_actual_2,
    'rmse': rmse_2,
    'r2_score': r2_score_2
}

# In[18]
# Combine all results
results = {
    'Region 0': results_region_0,
    'Region 1': results_region_1,
    'Region 2': results_region_2
}

print("\n Models successfully trained and tested for all 3 regions")

# In[19]
# Create visualization for predictions vs actual values
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (region_name, result) in enumerate(results.items()):
    ax = axes[idx]
    ax.scatter(result['actual'], result['predictions'], alpha=0.5)
    ax.plot([result['actual'].min(), result['actual'].max()], 
            [result['actual'].min(), result['actual'].max()], 
            'r--', lw=2)
    ax.set_xlabel('Actual Volume (thousand barrels)')
    ax.set_ylabel('Predicted Volume (thousand barrels)')
    ax.set_title(f'{region_name} - Predictions vs Actual')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# In[23]
# 3.1 Store key values for calculations
BUDGET = 100_000_000  # 100 million USD
REVENUE_PER_BARREL = 4.5  # USD per barrel
REVENUE_PER_UNIT = 4500  # USD per thousand barrels (unit)
NUM_WELLS_TO_STUDY = 500
NUM_WELLS_TO_DEVELOP = 200
COST_PER_WELL = BUDGET / NUM_WELLS_TO_DEVELOP  # 500,000 USD per well

print("Key Business Parameters:")
print(f"Budget: ${BUDGET:,}")
print(f"Revenue per barrel: ${REVENUE_PER_BARREL}")
print(f"Revenue per unit (thousand barrels): ${REVENUE_PER_UNIT:,}")
print(f"Wells to study: {NUM_WELLS_TO_STUDY}")
print(f"Wells to develop: {NUM_WELLS_TO_DEVELOP}")
print(f"Cost per well: ${COST_PER_WELL:,}")

# In[25]
# 3.2 Calculate breakeven volume
breakeven_volume = COST_PER_WELL / REVENUE_PER_UNIT
print(f"\nBreakeven volume per well: {breakeven_volume:.2f} thousand barrels")

# Compare with average volumes in each region
print("\nAverage actual volume by region:")
for region_name, data in regions.items():
    avg_volume = data[target].mean()
    print(f"{region_name}: {avg_volume:.2f} thousand barrels")
    if avg_volume > breakeven_volume:
        print(f"   Above breakeven (profit: ${(avg_volume - breakeven_volume) * REVENUE_PER_UNIT:,.2f})")
    else:
        print(f"   Below breakeven (loss: ${(breakeven_volume - avg_volume) * REVENUE_PER_UNIT:,.2f})")

# In[29]
def calculate_profit(predictions, actual, num_wells_to_develop=200):
    """
    Calculate profit from selected oil wells.
    
    Parameters:
    - predictions: array of predicted volumes
    - actual: array of actual volumes
    - num_wells_to_develop: number of wells to develop (default 200)
    
    Returns:
    - profit: total profit in USD
    - selected_wells: indices of selected wells
    - total_volume: total volume from selected wells
    """
    # 4.1 Pick wells with highest predicted values
    top_indices = np.argsort(predictions)[-num_wells_to_develop:]
    
    # 4.2 Get actual volumes for selected wells
    selected_actual_volumes = actual[top_indices]
    total_volume = selected_actual_volumes.sum()
    
    # 4.3 Calculate profit
    revenue = total_volume * REVENUE_PER_UNIT
    profit = revenue - BUDGET
    
    return profit, top_indices, total_volume

# Calculate profit for each region

print("PROFIT CALCULATION FOR EACH REGION")


# Calculate for Region 0
print("\nRegion 0:")
profit_0, selected_wells_0, total_volume_0 = calculate_profit(
    results['Region 0']['predictions'], 
    results['Region 0']['actual']
)
results['Region 0']['profit'] = profit_0
results['Region 0']['total_volume'] = total_volume_0

print(f"  Total volume from 200 wells: {total_volume_0:,.2f} thousand barrels")
print(f"  Total revenue: ${total_volume_0 * REVENUE_PER_UNIT:,.2f}")
print(f"  Total cost: ${BUDGET:,.2f}")
print(f"  Profit: ${profit_0:,.2f}")
print(f"  ROI: {(profit_0/BUDGET)*100:.2f}%")

# Calculate for Region 1
print("\nRegion 1:")
profit_1, selected_wells_1, total_volume_1 = calculate_profit(
    results['Region 1']['predictions'], 
    results['Region 1']['actual']
)
results['Region 1']['profit'] = profit_1
results['Region 1']['total_volume'] = total_volume_1

print(f"  Total volume from 200 wells: {total_volume_1:,.2f} thousand barrels")
print(f"  Total revenue: ${total_volume_1 * REVENUE_PER_UNIT:,.2f}")
print(f"  Total cost: ${BUDGET:,.2f}")
print(f"  Profit: ${profit_1:,.2f}")
print(f"  ROI: {(profit_1/BUDGET)*100:.2f}%")

# Calculate for Region 2
print("\nRegion 2:")
profit_2, selected_wells_2, total_volume_2 = calculate_profit(
    results['Region 2']['predictions'], 
    results['Region 2']['actual']
)
results['Region 2']['profit'] = profit_2
results['Region 2']['total_volume'] = total_volume_2

print(f"  Total volume from 200 wells: {total_volume_2:,.2f} thousand barrels")
print(f"  Total revenue: ${total_volume_2 * REVENUE_PER_UNIT:,.2f}")
print(f"  Total cost: ${BUDGET:,.2f}")
print(f"  Profit: ${profit_2:,.2f}")
print(f"  ROI: {(profit_2/BUDGET)*100:.2f}%")

# In[32]
def bootstrap_profit(predictions, actual, n_samples=1000, sample_size=500, n_develop=200):
    """
    Use bootstrapping to estimate profit distribution.
    """
    profits = []
    
    for i in range(n_samples):
        # Sample indices with replacement
        sample_indices = np.random.choice(len(predictions), size=sample_size, replace=True)
        
        # Get predictions and actual values for the sample
        sample_predictions = predictions[sample_indices]
        sample_actual = actual[sample_indices]
        
        # Calculate profit for this sample
        profit, _, _ = calculate_profit(sample_predictions, sample_actual, n_develop)
        profits.append(profit)
    
    return np.array(profits)

# 5.1 Apply bootstrapping to each region

print("BOOTSTRAPPING ANALYSIS (1000 samples):")


# Initialize bootstrap results dictionary
bootstrap_results = {}

# Bootstrap for Region 0
print("\nRegion 0:")


all_predictions_0 = results['Region 0']['model'].predict(region_0[features])
all_actual_0 = region_0[target].values

profits_0 = bootstrap_profit(all_predictions_0, all_actual_0)

avg_profit_0 = profits_0.mean()
std_profit_0 = profits_0.std()
confidence_interval_0 = np.percentile(profits_0, [2.5, 97.5])
risk_of_loss_0 = (profits_0 < 0).mean() * 100

bootstrap_results['Region 0'] = {
    'profits': profits_0,
    'avg_profit': avg_profit_0,
    'std_profit': std_profit_0,
    'confidence_interval': confidence_interval_0,
    'risk_of_loss': risk_of_loss_0
}


print(f"  Average profit: ${avg_profit_0:,.2f}")
print(f"  Standard deviation: ${std_profit_0:,.2f}")
print(f"  95% Confidence interval: [${confidence_interval_0[0]:,.2f}, ${confidence_interval_0[1]:,.2f}]")
print(f"  Risk of losses: {risk_of_loss_0:.2f}%")
if risk_of_loss_0 < 2.5:
    print(f"  Meets risk criteria (< 2.5%)")
else:
    print(f"  Does not meet risk criteria (>= 2.5%)")

# Bootstrap for Region 1
print("\nRegion 1:")


all_predictions_1 = results['Region 1']['model'].predict(region_1[features])
all_actual_1 = region_1[target].values

profits_1 = bootstrap_profit(all_predictions_1, all_actual_1)

avg_profit_1 = profits_1.mean()
std_profit_1 = profits_1.std()
confidence_interval_1 = np.percentile(profits_1, [2.5, 97.5])
risk_of_loss_1 = (profits_1 < 0).mean() * 100

bootstrap_results['Region 1'] = {
    'profits': profits_1,
    'avg_profit': avg_profit_1,
    'std_profit': std_profit_1,
    'confidence_interval': confidence_interval_1,
    'risk_of_loss': risk_of_loss_1
}


print(f"  Average profit: ${avg_profit_1:,.2f}")
print(f"  Standard deviation: ${std_profit_1:,.2f}")
print(f"  95% Confidence interval: [${confidence_interval_1[0]:,.2f}, ${confidence_interval_1[1]:,.2f}]")
print(f"  Risk of losses: {risk_of_loss_1:.2f}%")
if risk_of_loss_1 < 2.5:
    print(f"   Meets risk criteria (< 2.5%)")
else:
    print(f"   Does not meet risk criteria (>= 2.5%)")

# Bootstrap for Region 2
print("\nRegion 2:")


all_predictions_2 = results['Region 2']['model'].predict(region_2[features])
all_actual_2 = region_2[target].values

profits_2 = bootstrap_profit(all_predictions_2, all_actual_2)

avg_profit_2 = profits_2.mean()
std_profit_2 = profits_2.std()
confidence_interval_2 = np.percentile(profits_2, [2.5, 97.5])
risk_of_loss_2 = (profits_2 < 0).mean() * 100

bootstrap_results['Region 2'] = {
    'profits': profits_2,
    'avg_profit': avg_profit_2,
    'std_profit': std_profit_2,
    'confidence_interval': confidence_interval_2,
    'risk_of_loss': risk_of_loss_2
}


print(f"  Average profit: ${avg_profit_2:,.2f}")
print(f"  Standard deviation: ${std_profit_2:,.2f}")
print(f"  95% Confidence interval: [${confidence_interval_2[0]:,.2f}, ${confidence_interval_2[1]:,.2f}]")
print(f"  Risk of losses: {risk_of_loss_2:.2f}%")
if risk_of_loss_2 < 2.5:
    print(f"  Meets risk criteria (< 2.5%)")
else:
    print(f"  Does not meet risk criteria (>= 2.5%)")

print("\n Bootstrap analysis completed for all 3 regions")

# In[33]
# Create visualization for profit distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Region 0
ax0 = axes[0]
ax0.hist(bootstrap_results['Region 0']['profits'], bins=50, alpha=0.7, edgecolor='black')
ax0.axvline(0, color='red', linestyle='--', label='Breakeven', linewidth=2)
ax0.axvline(bootstrap_results['Region 0']['avg_profit'], color='green', 
            linestyle='-', linewidth=2, 
            label=f'Avg: ${bootstrap_results["Region 0"]["avg_profit"]/1e6:.1f}M')
ax0.set_xlabel('Profit (USD)')
ax0.set_ylabel('Frequency')
ax0.set_title(f'Region 0\nRisk: {bootstrap_results["Region 0"]["risk_of_loss"]:.2f}%')
ax0.legend()
ax0.grid(True, alpha=0.3)

# Plot Region 1
ax1 = axes[1]
ax1.hist(bootstrap_results['Region 1']['profits'], bins=50, alpha=0.7, edgecolor='black')
ax1.axvline(0, color='red', linestyle='--', label='Breakeven', linewidth=2)
ax1.axvline(bootstrap_results['Region 1']['avg_profit'], color='green', 
            linestyle='-', linewidth=2, 
            label=f'Avg: ${bootstrap_results["Region 1"]["avg_profit"]/1e6:.1f}M')
ax1.set_xlabel('Profit (USD)')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Region 1\nRisk: {bootstrap_results["Region 1"]["risk_of_loss"]:.2f}%')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Region 2
ax2 = axes[2]
ax2.hist(bootstrap_results['Region 2']['profits'], bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', label='Breakeven', linewidth=2)
ax2.axvline(bootstrap_results['Region 2']['avg_profit'], color='green', 
            linestyle='-', linewidth=2, 
            label=f'Avg: ${bootstrap_results["Region 2"]["avg_profit"]/1e6:.1f}M')
ax2.set_xlabel('Profit (USD)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Region 2\nRisk: {bootstrap_results["Region 2"]["risk_of_loss"]:.2f}%')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# In[35]

print("Final Check")

# Check which regions meet risk criteria
suitable_regions = []
for region_name, data in bootstrap_results.items():
    if data['risk_of_loss'] < 2.5:
        suitable_regions.append((region_name, data))

print(f"\nRegions meeting risk criteria (< 2.5% risk):")
if suitable_regions:
    for region_name, data in suitable_regions:
        print(f"   {region_name}: {data['risk_of_loss']:.2f}% risk")
else:
    print("   No regions meet the risk criteria")

# Show regions that don't meet criteria
print(f"\nRegions NOT meeting risk criteria:")
for region_name, data in bootstrap_results.items():
    if data['risk_of_loss'] >= 2.5:
        print(f"   {region_name}: {data['risk_of_loss']:.2f}% risk")

if suitable_regions:
    # Select region with highest average profit
    best_region = max(suitable_regions, key=lambda x: x[1]['avg_profit'])
    region_name, region_data = best_region

# In[38]
# Summary Statistics Table


# Create summary table

print("SUMMARY TABLE - ALL REGIONS")


summary_data = []
for region_name in ['Region 0', 'Region 1', 'Region 2']:
    summary_data.append({
        'Region': region_name,
        'Avg Volume': f"{results[region_name]['avg_actual']:.2f}",
        'RMSE': f"{results[region_name]['rmse']:.2f}",
        'R² Score': f"{results[region_name]['r2_score']:.4f}",
        'Single Profit': f"${results[region_name]['profit']/1e6:.2f}M",
        'Avg Profit': f"${bootstrap_results[region_name]['avg_profit']/1e6:.2f}M",
        'Risk': f"{bootstrap_results[region_name]['risk_of_loss']:.2f}%",
        'Meets Criteria': '✓' if bootstrap_results[region_name]['risk_of_loss'] < 2.5 else '✗'
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

