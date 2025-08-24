# In[3]
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# In[5]
# 1.1 Load the data files
print("\n1.1 Loading data files")
train_df = pd.read_csv('/datasets/gold_recovery_train.csv')
test_df = pd.read_csv('/datasets/gold_recovery_test.csv')
full_df = pd.read_csv('/datasets/gold_recovery_full.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Full dataset shape: {full_df.shape}")

# Display basic information about the datasets
print("\nFirst few rows of training data:")
display(train_df.head())

# In[6]
print("\nData types in training set:")
print(train_df.dtypes.value_counts())

print("\nBasic statistics of training data:")
display(train_df.describe())

# In[7]
# 1.2 Check recovery calculation
print("\n1.2 Check recovery calculation")

# Recovery formula: Recovery = C*(F-T)/(F*(C-T)) * 100%
# Where: C = concentrate, F = feed, T = tails

def calculate_recovery(c, f, t):
    """
    Calculate recovery using the formula: Recovery = C*(F-T)/(F*(C-T)) * 100%
    
    Parameters:
    c: concentrate grade
    f: feed grade
    t: tails grade
    
    Returns:
    recovery percentage
    """
    # Handle edge cases to avoid division by zero and invalid values
    if pd.isna(c) or pd.isna(f) or pd.isna(t):
        return np.nan
    
    if f == 0 or (c - t) == 0:
        return 0
    
    if c == t: 
        return 0
    
    try:
        recovery = (c * (f - t)) / (f * (c - t)) * 100
        # Check if result is valid
        if np.isfinite(recovery):
            return recovery
        else:
            return 0
    except:
        return 0

# Calculate rougher output recovery
c_rougher = train_df['rougher.output.concentrate_au'].values
f_rougher = train_df['rougher.input.feed_au'].values
t_rougher = train_df['rougher.output.tail_au'].values

# Calculate recovery for each row
calculated_recovery = []
for i in range(len(train_df)):
    recovery = calculate_recovery(c_rougher[i], f_rougher[i], t_rougher[i])
    calculated_recovery.append(recovery)

calculated_recovery = np.array(calculated_recovery)

# Get actual recovery values from the dataset
actual_recovery = train_df['rougher.output.recovery'].values

# Remove NaN values for comparison
mask = ~(np.isnan(calculated_recovery) | np.isnan(actual_recovery))
calculated_recovery_clean = calculated_recovery[mask]
actual_recovery_clean = actual_recovery[mask]

print(f"Total samples: {len(train_df)}")
print(f"Valid samples for comparison: {len(actual_recovery_clean)}")

# Calculate MAE
mae_recovery = mean_absolute_error(actual_recovery_clean, calculated_recovery_clean)
print(f"\nMAE between calculated and actual recovery: {mae_recovery:.4f}")

# Calculated vs Actual Findings
print("\nComparison of calculated vs actual recovery Findings (first 5 valid rows):")
valid_indices = np.where(mask)[0][:5]
comparison_df = pd.DataFrame({
    'Actual': actual_recovery[valid_indices],
    'Calculated': calculated_recovery[valid_indices],
    'Difference': abs(actual_recovery[valid_indices] - calculated_recovery[valid_indices])
})
display(comparison_df)

# In[8]
# 1.3 Analyze features not available in test set
print("\n1.3 Analyzing features not available in test set")

train_columns = set(train_df.columns)
test_columns = set(test_df.columns)
missing_in_test = train_columns - test_columns

print(f"\nNumber of features in training set: {len(train_columns)}")
print(f"Number of features in test set: {len(test_columns)}")
print(f"Number of features missing in test set: {len(missing_in_test)}")

# In[9]
print("\nFeatures missing in test set:")
missing_features = sorted(list(missing_in_test))
for feature in missing_features:
    print(f"  - {feature}")

# In[10]
# Analyze the type of missing features
print("\nAnalysis of missing features:")
output_features = [f for f in missing_features if 'output' in f]
calc_features = [f for f in missing_features if 'calc' in f]

print(f"Output features (measured after process): {len(output_features)}")
print(f"Calculated features: {len(calc_features)}")

# In[12]
# 1.4 Data preprocessing
print("\n1.4 Performing data preprocessing")

# Check for missing values in training set
print("\nMissing values in training set:")
missing_train = train_df.isnull().sum()
display(missing_train[missing_train > 0])

# In[13]
# Check for missing values in training set
print("\nMissing values in test set:")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])

# In[14]
# Define target columns
target_cols = ['rougher.output.recovery', 'final.output.recovery']

# Remove rows where target columns have missing values (training set only)
train_df_no_missing_targets = train_df.dropna(subset=target_cols)
print(f"Rows removed due to missing targets: {len(train_df) - len(train_df_no_missing_targets)}")

# Fill missing values with median (only for features, not targets)
train_df_clean = train_df_no_missing_targets.fillna(train_df_no_missing_targets.median(numeric_only=True))
test_df_clean = test_df.fillna(test_df.median(numeric_only=True))

print("\nMissing values after preprocessing:")
print(f"Training set: {train_df_clean.isnull().sum().sum()}")
print(f"Test set: {test_df_clean.isnull().sum().sum()}")

# In[18]
# 2.1 Metal concentrations at different stages
print("\n2.1 Analyzing metal concentrations at different purification stages")

# Define the stages and metals
stages = ['rougher', 'primary_cleaner', 'secondary_cleaner', 'final']
metals = ['au', 'ag', 'pb']

# Create a figure for visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, metal in enumerate(metals):
    # Collect concentration data for each stage
    stage_data = []
    stage_names = []
    
    # Input feed
    if f'rougher.input.feed_{metal}' in train_df_clean.columns:
        stage_data.append(train_df_clean[f'rougher.input.feed_{metal}'].mean())
        stage_names.append('Feed')
    
    # Rougher concentrate
    if f'rougher.output.concentrate_{metal}' in train_df_clean.columns:
        stage_data.append(train_df_clean[f'rougher.output.concentrate_{metal}'].mean())
        stage_names.append('Rougher')
    
    # Primary cleaner
    if f'primary_cleaner.output.concentrate_{metal}' in train_df_clean.columns:
        stage_data.append(train_df_clean[f'primary_cleaner.output.concentrate_{metal}'].mean())
        stage_names.append('Primary')
    
    # Secondary cleaner
    if f'secondary_cleaner.output.concentrate_{metal}' in train_df_clean.columns:
        stage_data.append(train_df_clean[f'secondary_cleaner.output.concentrate_{metal}'].mean())
        stage_names.append('Secondary')
    
    # Final concentrate
    if f'final.output.concentrate_{metal}' in train_df_clean.columns:
        stage_data.append(train_df_clean[f'final.output.concentrate_{metal}'].mean())
        stage_names.append('Final')
    
    # Plot
    axes[idx].plot(stage_names, stage_data, marker='o', linewidth=2, markersize=8)
    axes[idx].set_title(f'{metal.upper()} Concentration by Stage')
    axes[idx].set_xlabel('Purification Stage')
    axes[idx].set_ylabel('Average Concentration')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# In[19]
# Create histograms for metal concentrations at different stages
print("\nCreating histograms for metal concentration distributions.")

# Define the stages to analyze
stages_to_plot = [
    ('rougher.input.feed_', 'Feed'),
    ('rougher.output.concentrate_', 'Rougher Concentrate'),
    ('primary_cleaner.output.concentrate_', 'Primary Cleaner'),
    ('final.output.concentrate_', 'Final Concentrate')
]

# Create histograms for each metal
for metal in metals:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'{metal.upper()} Concentration Distribution Across Stages', fontsize=16)
    
    for idx, (stage_prefix, stage_name) in enumerate(stages_to_plot):
        col_name = f'{stage_prefix}{metal}'
        
        if col_name in train_df_clean.columns:
            # Create histogram
            train_df_clean[col_name].hist(ax=axes[idx], bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(stage_name)
            axes[idx].set_xlabel(f'{metal.upper()} Concentration')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = train_df_clean[col_name].mean()
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_val:.2f}')
            axes[idx].legend()
        else:
            axes[idx].text(0.5, 0.5, 'Data not available', 
                         ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(stage_name)
    
    plt.tight_layout()
    plt.show()

# In[20]
# numerical analysis
print("\nAverage metal concentrations by stage:")
for metal in metals:
    print(f"\n{metal.upper()}:")
    if f'rougher.input.feed_{metal}' in train_df_clean.columns:
        print(f"  Feed: {train_df_clean[f'rougher.input.feed_{metal}'].mean():.2f}")
    if f'rougher.output.concentrate_{metal}' in train_df_clean.columns:
        print(f"  Rougher: {train_df_clean[f'rougher.output.concentrate_{metal}'].mean():.2f}")
    if f'final.output.concentrate_{metal}' in train_df_clean.columns:
        print(f"  Final: {train_df_clean[f'final.output.concentrate_{metal}'].mean():.2f}")

        # Initialize feed_size_cols globally
feed_size_cols = []

# In[21]
# 2.2 Compare feed particle size distributions
print("\n2.2 Comparing feed particle size distributions...")

# Debug: Let's see what columns we have
print("\nChecking available columns...")
all_cols = list(train_df_clean.columns)
print(f"Total columns: {len(all_cols)}")

# Look for any size-related columns
size_keywords = ['size', 'particle', 'granulometry', 'mesh', 'fraction']
potential_size_cols = []
for col in all_cols:
    for keyword in size_keywords:
        if keyword in col.lower():
            potential_size_cols.append(col)
            break

print(f"\nPotential size-related columns found: {len(potential_size_cols)}")
if len(potential_size_cols) > 0:
    print("Examples:", potential_size_cols[:5])

# Get feed size columns - try different patterns
feed_size_cols = [col for col in train_df_clean.columns if 'rougher.input.feed_size' in col]
if len(feed_size_cols) == 0:
    # Try alternative pattern
    feed_size_cols = [col for col in train_df_clean.columns if 'feed_size' in col]
if len(feed_size_cols) == 0:
    # Try another pattern
    feed_size_cols = [col for col in train_df_clean.columns if 'size' in col and 'input' in col]
    
print(f"Feed size columns found: {len(feed_size_cols)}")
if len(feed_size_cols) > 0:
    print("Columns:", feed_size_cols[:5], "..." if len(feed_size_cols) > 5 else "")
    
    # Calculate distributions
    train_size_dist = train_df_clean[feed_size_cols].mean()
    test_size_dist = test_df_clean[feed_size_cols].mean()
    
    # Visualize distributions
    plt.figure(figsize=(10, 6))
    x = range(len(feed_size_cols))
    plt.plot(x, train_size_dist.values, 'b-', label='Training Set', linewidth=2, marker='o', markersize=4)
    plt.plot(x, test_size_dist.values, 'r--', label='Test Set', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Size Fraction Index')
    plt.ylabel('Average Percentage')
    plt.title('Feed Particle Size Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x[::2])  # Show every other tick to avoid crowding
    plt.show()
    
    # Calculate difference
    size_diff = abs(train_size_dist - test_size_dist).mean()
    print(f"\nAverage absolute difference between distributions: {size_diff:.4f}")
    print("Note: Small differences indicate similar distributions, which is good for model reliability.")
    
    # Show detailed statistics
    print("\nDistribution statistics:")
    print(f"Training set - Min: {train_size_dist.min():.2f}, Max: {train_size_dist.max():.2f}, Mean: {train_size_dist.mean():.2f}")
    print(f"Test set - Min: {test_size_dist.min():.2f}, Max: {test_size_dist.max():.2f}, Mean: {test_size_dist.mean():.2f}")
else:
    print("\nNo feed size columns found in the data.")
    print("Checking column names for size-related features...")
    size_related = [col for col in train_df_clean.columns if 'size' in col.lower()]
    print(f"Found {len(size_related)} size-related columns:")
    for col in size_related[:10]:  # Show first 10
        print(f"  - {col}")
    
    # Set a default value for size_diff to avoid errors later
    size_diff = 0.0
    print("\nSkipping particle size distribution comparison.")

# In[22]
# Create overlaid histograms for better distribution comparison
if len(feed_size_cols) == 1:
    # Single column - create one histogram
    plt.figure(figsize=(10, 6))
    plt.hist(train_df_clean[feed_size_cols[0]], bins=50, alpha=0.6, label='Training Set', 
                color='blue', edgecolor='black')
    plt.hist(test_df_clean[feed_size_cols[0]], bins=50, alpha=0.6, label='Test Set', 
                color='red', edgecolor='black')
    plt.xlabel('Feed Size Value')
    plt.ylabel('Frequency')
    plt.title('Feed Particle Size Distribution - Histogram Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    # Multiple columns - create subplots
    n_cols = min(len(feed_size_cols), 4)  # Show max 4 columns per row
    n_rows = (len(feed_size_cols) + n_cols - 1) // n_cols
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
    for idx, col in enumerate(feed_size_cols[:n_rows*n_cols]):
        ax = axes[idx]
        # Overlay histograms
        ax.hist(train_df_clean[col], bins=30, alpha=0.6, label='Training', 
                color='blue', edgecolor='black')
        ax.hist(test_df_clean[col], bins=30, alpha=0.6, label='Test', 
                color='red', edgecolor='black')
        ax.set_title(f'Size Fraction {idx}')
        ax.set_xlabel('Percentage')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Hide empty subplots
    for idx in range(len(feed_size_cols), len(axes)):
        axes[idx].set_visible(False)
        
    plt.suptitle('Feed Particle Size Distributions - Histogram Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

# In[24]
# 2.3 Total concentrations analysis

# Calculate total concentrations for different stages
def calculate_total_concentration(df, stage_prefix):
    """Calculate total concentration for a given stage"""
    # Get all concentration columns for the stage
    cols = [col for col in df.columns if stage_prefix in col and 
            any(metal in col for metal in ['_au', '_ag', '_pb', '_sol'])]
    
    if len(cols) > 0:
        return df[cols].sum(axis=1)
    else:
        return pd.Series([0] * len(df))

# Calculate totals for different stages
train_df_clean['total_feed'] = calculate_total_concentration(
    train_df_clean, 'rougher.input.feed'
)
train_df_clean['total_rougher'] = calculate_total_concentration(
    train_df_clean, 'rougher.output.concentrate'
)
train_df_clean['total_final'] = calculate_total_concentration(
    train_df_clean, 'final.output.concentrate'
)
print("\n2.3 Total concentrations analyzed at different stages!")

# In[25]
# Visualize distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

stages_to_plot = [
    ('total_feed', 'Raw Feed'),
    ('total_rougher', 'Rougher Concentrate'),
    ('total_final', 'Final Concentrate')
]

for idx, (col, title) in enumerate(stages_to_plot):
    data = train_df_clean[col]
    axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{title} Total Concentration')
    axes[idx].set_xlabel('Total Concentration')
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# In[26]
# Check for anomalies (values > 100 or < 0)
print("\nChecking for anomalous total concentrations")
for col, title in stages_to_plot:
    anomalies = train_df_clean[(train_df_clean[col] > 100) | (train_df_clean[col] < 0)]
    print(f"\n{title}:")
    print(f"  Total samples: {len(train_df_clean)}")
    print(f"  Anomalous samples (>100 or <0): {len(anomalies)}")
    print(f"  Percentage: {len(anomalies)/len(train_df_clean)*100:.2f}%")

# In[27]
# Remove anomalies
print("\nRemoving anomalous samples")
initial_size = len(train_df_clean)


# Keep only samples where total concentrations are reasonable (0-100)
mask = (
    (train_df_clean['total_feed'] >= 2) & (train_df_clean['total_feed'] <= 100) &
    (train_df_clean['total_rougher'] >= 2) & (train_df_clean['total_rougher'] <= 100) &
    (train_df_clean['total_final'] >= 2) & (train_df_clean['total_final'] <= 100)
)
train_df_final = train_df_clean[mask].copy()

print(f"Samples before cleaning: {initial_size}")
print(f"Samples after cleaning: {len(train_df_final)}")
print(f"Samples removed: {initial_size - len(train_df_final)} ({(initial_size - len(train_df_final))/initial_size*100:.2f}%)")

# In[30]
# Building the Model

# 3.1 Define sMAPE function
print("\n3.1 Defining sMAPE evaluation metric")

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    
    Formula: sMAPE = (1/n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)) * 100
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    mask = denominator != 0
    smape_value = np.zeros_like(y_true)
    smape_value[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return 100.0 * smape_value.mean()

def final_smape(y_true, y_pred):
    """
    Calculate final sMAPE as weighted average of two targets
    
    Weights: 25% for rougher recovery, 75% for final recovery
    """
    # Assuming y_true and y_pred have two columns: [rougher_recovery, final_recovery]
    if len(y_true.shape) == 1:  # Single target
        return smape(y_true, y_pred)
    
    smape_rougher = smape(y_true[:, 0], y_pred[:, 0])
    smape_final = smape(y_true[:, 1], y_pred[:, 1])
    
    return 0.25 * smape_rougher + 0.75 * smape_final

# Create scorer for sklearn
smape_scorer = make_scorer(smape, greater_is_better=False)

# In[32]
# Prepare features and targets

# Define target columns
target_cols = ['rougher.output.recovery', 'final.output.recovery']
# Check if target columns exist
print(f"\nChecking for required columns")

# In[33]
# First check if train_df_final exists
if 'train_df_final' not in locals():
    print("ERROR: train_df_final doesn't exist! Cannot proceed.")
    print("Check if anomaly removal in step 2.3 removed all data.")
    # Create empty variables to prevent later errors
    X_train_scaled = np.array([])
    y_train_rougher = pd.Series([])
    y_train_final = pd.Series([])
    X_test_scaled = np.array([])
else:
    print(f" train_df_final exists with {len(train_df_final)} samples")

# In[34]
# Check for target columns
missing_targets = []
for col in target_cols:
    if col in train_df_final.columns:
        print(f" {col} found")
    else:
        print(f" {col} NOT FOUND")
        missing_targets.append(col)
    
if len(missing_targets) > 0:
    print(f"\nERROR: Missing target columns: {missing_targets}")
    print("Cannot proceed with model training without target variables!")
    # Create empty variables
    X_train_scaled = np.array([])
    y_train_rougher = pd.Series([])
    y_train_final = pd.Series([])
    X_test_scaled = np.array([])
else:
    # Get feature columns (exclude targets and columns not in test set)
    feature_cols = [col for col in train_df_final.columns 
                    if col in test_df_clean.columns and col != 'date']
        
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Target variables: {target_cols}")
        
    # Show some feature examples
    print(f"\nExample features: {feature_cols[:5]}...")

# In[35]
# Prepare data
try:
    X_train = train_df_final[feature_cols]
    y_train_rougher = train_df_final[target_cols[0]]
    y_train_final = train_df_final[target_cols[1]]
    X_test = test_df_clean[feature_cols]
            
    print(f"\nData shapes before scaling:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train_rougher: {y_train_rougher.shape}")
    print(f"  y_train_final: {y_train_final.shape}")
    print(f"  X_test: {X_test.shape}")
            
    # Scale features
    print("\nScaling features:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
            
    print("Feature scaling completed successfully")
    print(f"\nFinal data shapes:")
    print(f"  X_train_scaled: {X_train_scaled.shape}")
    print(f"  X_test_scaled: {X_test_scaled.shape}")
            
    print("\nData preparation complete!")
            
except Exception as e:
    print(f"\nERROR during data preparation: {str(e)}")
    print("Creating empty variables to continue")
    X_train_scaled = np.array([])
    y_train_rougher = pd.Series([])
    y_train_final = pd.Series([])
    X_test_scaled = np.array([])

# In[36]

print("DATA PREPARATION SUMMARY")

print(f"X_train_scaled exists: {'X_train_scaled' in locals() and len(X_train_scaled) > 0}")
print(f"y_train_rougher exists: {'y_train_rougher' in locals() and len(y_train_rougher) > 0}")
print(f"y_train_final exists: {'y_train_final' in locals() and len(y_train_final) > 0}")
print(f"X_test_scaled exists: {'X_test_scaled' in locals() and len(X_test_scaled) > 0}")
print("-"*60)

# In[38]
# 3.3 Train and evaluate models (This is step 3.2 in the instructions)
print("\n3.3 Training and evaluating models")
print("="*60)

# Import required library
from sklearn.model_selection import cross_val_score

# Define models to test
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
}

print(f"Models to evaluate: {list(models.keys())}")

# In[39]
# Check if we have the necessary data
required_vars = {
    'X_train_scaled': 'X_train_scaled' in locals() and len(X_train_scaled) > 0,
    'y_train_rougher': 'y_train_rougher' in locals() and len(y_train_rougher) > 0,
    'y_train_final': 'y_train_final' in locals() and len(y_train_final) > 0,
    'X_test_scaled': 'X_test_scaled' in locals() and len(X_test_scaled) > 0
}

print("Checking required variables:")
for var_name, exists in required_vars.items():
    print(f"  {var_name}: {' Found' if exists else ' Not found'}")

if not all(required_vars.values()):
    print("\nERROR: Missing required data for model training!")
    print("Cannot proceed - check previous steps.")
else:
    print(f"\n All required data found!")
    print(f"Training data: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")

# In[40]
# Initialize results dictionary
results = {}

# Only run if we have all required data
if all(required_vars.values()):
    # Train and evaluate each model
    for model_num, (name, model) in enumerate(models.items(), 1):
        print(f"\n{'='*40}")
        print(f"Model {model_num}/{len(models)}: {name}")
        print(f"{'='*40}")
        
        try:
            # Cross-validate for rougher recovery
            print(f"Cross-validating for rougher recovery...")
            cv_scores_rougher = cross_val_score(
                model, X_train_scaled, y_train_rougher, 
                cv=3, scoring=smape_scorer  # Reduced from 5 to 3 folds
            )
            rougher_smape = -cv_scores_rougher.mean()
            
            # Cross-validate for final recovery
            print(f"Cross-validating for final recovery...")
            cv_scores_final = cross_val_score(
                model, X_train_scaled, y_train_final, 
                cv=3, scoring=smape_scorer  # Reduced from 5 to 3 folds
            )
            final_smape = -cv_scores_final.mean()
            
            # Calculate weighted total
            total_smape = 0.25 * rougher_smape + 0.75 * final_smape
            
            # Store results
            results[name] = {
                'rougher_smape': rougher_smape,
                'final_smape': final_smape,
                'total_smape': total_smape
            }
            
            # Print results
            print(f"  Rougher Recovery sMAPE: {rougher_smape:.4f}")
            print(f"  Final Recovery sMAPE: {final_smape:.4f}")
            print(f"  Total sMAPE (weighted): {total_smape:.4f}")
            
        except Exception as e:
            print(f"ERROR training {name}: {str(e)}")
else:
    print("Skipping model training due to missing data.")

# In[41]
    # Display summary
    print(f"\n{'='*40}")
    print("MODEL COMPARISON SUMMARY")
    print(f"\n{'='*40}")
    
    if len(results) > 0:
        # Create summary DataFrame for easy viewing
        summary_df = pd.DataFrame(results).T
        print(summary_df.round(4))
        
        # Select best model
        best_model_name = min(results, key=lambda x: results[x]['total_smape'])
        print(f"\nBest model: {best_model_name}")
        print(f"Best total sMAPE: {results[best_model_name]['total_smape']:.4f}")
        
        # Train best model on full data
        print(f"\nTraining {best_model_name} on full training data")
        best_model = models[best_model_name]
        
        # Train separate models for each target
        best_model_rougher = best_model.__class__(**best_model.get_params())
        best_model_final = best_model.__class__(**best_model.get_params())
        
        best_model_rougher.fit(X_train_scaled, y_train_rougher)
        best_model_final.fit(X_train_scaled, y_train_final)
        
        # Make predictions on test set
        print("\nMaking predictions on test set...")
        test_pred_rougher = best_model_rougher.predict(X_test_scaled)
        test_pred_final = best_model_final.predict(X_test_scaled)
        
        print(f" Predictions completed")
        print(f"  Test predictions shape: ({len(test_pred_rougher)},)")
    else:
        print("No models were successfully trained!")

# In[42]
# Train the best model on full data and make predictions
if len(results) > 0 and all(required_vars.values()):
    print(f"\nTraining {best_model_name} on full training data")
    best_model = models[best_model_name]
    
    try:
        # Train separate models for each target
        best_model_rougher = best_model.__class__(**best_model.get_params())
        best_model_final = best_model.__class__(**best_model.get_params())
        
        best_model_rougher.fit(X_train_scaled, y_train_rougher)
        best_model_final.fit(X_train_scaled, y_train_final)
        print("Models trained successfully")
        
        # Show model details
        print(f"\nModel details:")
        print(f"  Model type: {best_model.__class__.__name__}")
        if hasattr(best_model, 'get_params'):
            params = best_model.get_params()
            print(f"  Parameters: {params}")
        
    except Exception as e:
        print(f"ERROR training final models: {str(e)}")

# In[43]
# Make predictions on test set
if 'best_model_rougher' in locals() and 'best_model_final' in locals():
    try:
        print("\nMaking predictions on test set")
        test_pred_rougher = best_model_rougher.predict(X_test_scaled)
        test_pred_final = best_model_final.predict(X_test_scaled)
        
        print(f" Predictions completed")
        print(f"  Rougher predictions: {test_pred_rougher.shape}")
        print(f"  Final predictions: {test_pred_final.shape}")
        print(f"\nPrediction statistics:")
        print(f"  Rougher - Mean: {test_pred_rougher.mean():.2f}, Std: {test_pred_rougher.std():.2f}")
        print(f"  Final - Mean: {test_pred_final.mean():.2f}, Std: {test_pred_final.std():.2f}")
        print(f"\nSample predictions (first 5):")
        print(f"  Rougher: {test_pred_rougher[:5].round(2)}")
        print(f"  Final: {test_pred_final[:5].round(2)}")
        
    except Exception as e:
        print(f"ERROR making predictions: {str(e)}")
else:
    print("Best models not available for predictions.")

# In[44]
# Visualize model performance comparison
if len(results) > 0:
    import matplotlib.pyplot as plt
    
    # Create bar plot of model performance
    plt.figure(figsize=(10, 6))
    
    models_names = list(results.keys())
    rougher_scores = [results[m]['rougher_smape'] for m in models_names]
    final_scores = [results[m]['final_smape'] for m in models_names]
    total_scores = [results[m]['total_smape'] for m in models_names]
    
    x = np.arange(len(models_names))
    width = 0.25
    
    plt.bar(x - width, rougher_scores, width, label='Rougher sMAPE', alpha=0.8)
    plt.bar(x, final_scores, width, label='Final sMAPE', alpha=0.8)
    plt.bar(x + width, total_scores, width, label='Total sMAPE', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('sMAPE (%)')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(total_scores):
        plt.text(i + width, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

