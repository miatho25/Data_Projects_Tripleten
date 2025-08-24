# In[3]
#import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils import resample

# In[4]
# Load the data
data = pd.read_csv('/datasets/Churn.csv')

# Display basic information
print("Dataset shape:", data.shape)
print("\nData info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nDescriptive statistics:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())

# In[7]
# Remove unnecessary columns
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
existing_columns = [col for col in columns_to_drop if col in data.columns]
data = data.drop(existing_columns, axis=1)
print(f"Dropped columns: {existing_columns}")

# **NEW: Check for duplicates after removing identifier columns**
print(f"\nChecking for duplicates after removing identifier columns:")
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
   print("Removing duplicate rows...")
   data = data.drop_duplicates()
   print(f"Dataset shape after removing duplicates: {data.shape}")
else:
   print("No duplicate rows found.")

# **NEW: Data Distribution Analysis with Visualizations**
print("\n" + "="*50)
print("DATA DISTRIBUTION ANALYSIS")
print("="*50)

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribution Analysis of Key Features', fontsize=16, fontweight='bold')

# Analyze numerical features
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']

# 1. Credit Score Distribution
axes[0,0].hist(data['CreditScore'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0,0].set_title('Credit Score Distribution', fontweight='bold')
axes[0,0].set_xlabel('Credit Score')
axes[0,0].set_ylabel('Frequency')
axes[0,0].grid(True, alpha=0.3)

# 2. Age Distribution  
axes[0,1].hist(data['Age'], bins=25, color='lightcoral', alpha=0.7, edgecolor='black')
axes[0,1].set_title('Customer Age Distribution', fontweight='bold')
axes[0,1].set_xlabel('Age (years)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].grid(True, alpha=0.3)

# 3. Tenure Distribution
axes[0,2].hist(data['Tenure'].dropna(), bins=11, color='lightgreen', alpha=0.7, edgecolor='black')
axes[0,2].set_title('Customer Tenure Distribution', fontweight='bold')
axes[0,2].set_xlabel('Tenure (years)')
axes[0,2].set_ylabel('Frequency')
axes[0,2].grid(True, alpha=0.3)

# 4. Balance Distribution (with log scale due to many zeros)
axes[1,0].hist(data['Balance'], bins=50, color='gold', alpha=0.7, edgecolor='black')
axes[1,0].set_title('Account Balance Distribution', fontweight='bold')
axes[1,0].set_xlabel('Balance ($)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].grid(True, alpha=0.3)

# 5. Estimated Salary Distribution
axes[1,1].hist(data['EstimatedSalary'], bins=30, color='plum', alpha=0.7, edgecolor='black')
axes[1,1].set_title('Estimated Salary Distribution', fontweight='bold')
axes[1,1].set_xlabel('Estimated Salary ($)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].grid(True, alpha=0.3)

# 6. Number of Products
product_counts = data['NumOfProducts'].value_counts().sort_index()
axes[1,2].bar(product_counts.index, product_counts.values, color='orange', alpha=0.7, edgecolor='black')
axes[1,2].set_title('Number of Banking Products', fontweight='bold')
axes[1,2].set_xlabel('Number of Products')
axes[1,2].set_ylabel('Number of Customers')
axes[1,2].grid(True, alpha=0.3)

# 7. Geography Distribution
geo_counts = data['Geography'].value_counts()
axes[2,0].bar(geo_counts.index, geo_counts.values, color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.7, edgecolor='black')
axes[2,0].set_title('Customer Geography Distribution', fontweight='bold')
axes[2,0].set_xlabel('Country')
axes[2,0].set_ylabel('Number of Customers')
axes[2,0].tick_params(axis='x', rotation=45)
axes[2,0].grid(True, alpha=0.3)

# 8. Gender Distribution
gender_counts = data['Gender'].value_counts()
axes[2,1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
             colors=['#ff9999', '#66b3ff'], startangle=90)
axes[2,1].set_title('Customer Gender Distribution', fontweight='bold')

# 9. Box plot for detecting outliers in key numerical features
data_for_boxplot = data[['Age', 'CreditScore', 'Balance', 'EstimatedSalary']].copy()
# Normalize for better visualization
for col in data_for_boxplot.columns:
   data_for_boxplot[col] = (data_for_boxplot[col] - data_for_boxplot[col].mean()) / data_for_boxplot[col].std()

axes[2,2].boxplot([data_for_boxplot[col].dropna() for col in data_for_boxplot.columns], 
                 labels=data_for_boxplot.columns)
axes[2,2].set_title('Outlier Detection (Standardized)', fontweight='bold')
axes[2,2].set_ylabel('Standardized Values')
axes[2,2].tick_params(axis='x', rotation=45)
axes[2,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# **Conclusions from Distribution Analysis**
print("\n" + "="*50)
print("KEY INSIGHTS FROM DISTRIBUTION ANALYSIS")
print("="*50)

print("""
CREDIT SCORE: 
  - Approximately normal distribution centered around 650
  - Range: 350-850, which is typical for credit scores
  - No significant outliers detected

AGE:
  - Right-skewed distribution with most customers aged 25-45
  - Few elderly customers (>65), suggesting target demographic
  - Some potential outliers in the 80+ range

TENURE:
  - Relatively uniform distribution from 0-10 years
  - Missing values present (909 missing) - will need imputation
  - Peak at lower tenure values suggests recent customer acquisition

BALANCE:
  - Heavily right-skewed with many zero balances
  - Large concentration of customers with $0 balance
  - High-value outliers present (>$200K) but appear legitimate

ESTIMATED SALARY:
  - Approximately uniform distribution from $11K to $200K
  - Good spread across income levels
  - No significant outliers detected

PRODUCTS:
  - Most customers have 1-2 banking products
  - Very few customers have 3-4 products
  - Potential churn indicator: customers with 1 product may be less engaged

GEOGRAPHY:
  - France has the highest customer base (~50%)
  - Germany and Spain are roughly equal (~25% each)
  - Geographic distribution should be considered in modeling

GENDER:
  - Relatively balanced gender distribution (slight female majority)
  - No significant gender bias in the dataset
""")

# In[8]
# Remove unnecessary columns
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
existing_columns = [col for col in columns_to_drop if col in data.columns]
data = data.drop(existing_columns, axis=1)
print(f"Dropped columns: {existing_columns}")

# Check for categorical features and encode them
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical features:", categorical_features)

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)
print("\nEncoded data shape:", data_encoded.shape)
print(data_encoded.columns)

# Split into features and target
X = data_encoded.drop('Exited', axis=1)
y = data_encoded['Exited']

# Split the data into training, validation, and test sets (60%, 20%, 20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Scale numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()

# Create copies of the datasets
X_train = X_train.copy()
X_val = X_val.copy() 
X_test = X_test.copy()

# Fit scaler on training data and transform all datasets
scaler.fit(X_train[numeric_features])
X_train.loc[:, numeric_features] = scaler.transform(X_train[numeric_features])
X_val.loc[:, numeric_features] = scaler.transform(X_val[numeric_features])
X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])

# In[10]
# Check class distribution
print("\nClass distribution:")
print(y.value_counts())
print("Class distribution percentage:")
print(y.value_counts(normalize=True) * 100)

# Visualize class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title('Class Distribution (0: Stayed, 1: Exited)')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# In[13]
# First lets check for missing values
# Check for missing values in the data
print("Missing values in X_train:")
print(X_train.isnull().sum())

# Check for infinity values or extremely large values
print("\nInfinity or extremely large values in X_train:")
print(np.isinf(X_train).sum())
print("\nMin and max values in X_train:")
print(X_train.min())
print(X_train.max())

# Handle missing or invalid values
# 1. Fill missing values
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_train.mean())  # Use training mean to avoid data leakage
X_test = X_test.fillna(X_train.mean())

# 2. Replace infinities with large finite values
X_train = X_train.replace([np.inf, -np.inf], np.finfo(np.float32).max)
X_val = X_val.replace([np.inf, -np.inf], np.finfo(np.float32).max)
X_test = X_test.replace([np.inf, -np.inf], np.finfo(np.float32).max)

# Verify the fix
print("\nAfter fixing, any missing values in X_train?")
print(X_train.isnull().sum().sum())
print("After fixing, any infinity values in X_train?")
print(np.isinf(X_train).sum().sum())

# In[14]
# Train a baseline Random Forest model
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = baseline_model.predict(X_val)
baseline_f1 = f1_score(y_val, y_val_pred)
baseline_auc = roc_auc_score(y_val, baseline_model.predict_proba(X_val)[:, 1])

print("\nBaseline model results (without addressing imbalance):")
print(f"F1 Score: {baseline_f1:.4f}")
print(f"AUC-ROC: {baseline_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# In[16]
# Approach 1: Class Weights:

# Train a Random Forest model with class weights
weighted_model = RandomForestClassifier(random_state=42, class_weight='balanced')
weighted_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred_weighted = weighted_model.predict(X_val)
weighted_f1 = f1_score(y_val, y_val_pred_weighted)
weighted_auc = roc_auc_score(y_val, weighted_model.predict_proba(X_val)[:, 1])

print("\nWeighted model results:")
print(f"F1 Score: {weighted_f1:.4f}")
print(f"AUC-ROC: {weighted_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_weighted))

# In[17]
# Approach 2: Upsampling the Minority Class

# First, separate majority and minority classes
X_train_majority = X_train[y_train == 0]
X_train_minority = X_train[y_train == 1]
y_train_majority = y_train[y_train == 0]
y_train_minority = y_train[y_train == 1]

# Upsample minority class
X_train_minority_upsampled, y_train_minority_upsampled = resample(
    X_train_minority, 
    y_train_minority,
    replace=True,
    n_samples=len(X_train_majority),
    random_state=42
)

# Combine majority class with upsampled minority class
X_train_upsampled = pd.concat([X_train_majority, X_train_minority_upsampled])
y_train_upsampled = pd.concat([y_train_majority, y_train_minority_upsampled])

print("\nClass distribution after upsampling:")
print(y_train_upsampled.value_counts())

# Train a model on upsampled data
upsampled_model = RandomForestClassifier(random_state=42)
upsampled_model.fit(X_train_upsampled, y_train_upsampled)

# Evaluate on validation set
y_val_pred_upsampled = upsampled_model.predict(X_val)
upsampled_f1 = f1_score(y_val, y_val_pred_upsampled)
upsampled_auc = roc_auc_score(y_val, upsampled_model.predict_proba(X_val)[:, 1])

print("\nUpsampling model results:")
print(f"F1 Score: {upsampled_f1:.4f}")
print(f"AUC-ROC: {upsampled_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_upsampled))

# In[18]
# Approach 3: Downsampling the Majority Class

# Downsample the majority class (Exited=0)
X_train_majority_downsampled, y_train_majority_downsampled = resample(
    X_train_majority, 
    y_train_majority,
    replace=False,
    n_samples=len(X_train_minority),
    random_state=42
)

# Combine downsampled majority class with minority class
X_train_downsampled = pd.concat([X_train_majority_downsampled, X_train_minority])
y_train_downsampled = pd.concat([y_train_majority_downsampled, y_train_minority])

print("\nClass distribution after downsampling:")
print(y_train_downsampled.value_counts())

# Train a model on downsampled data
downsampled_model = RandomForestClassifier(random_state=42)
downsampled_model.fit(X_train_downsampled, y_train_downsampled)

# Evaluate on validation set
y_val_pred_downsampled = downsampled_model.predict(X_val)
downsampled_f1 = f1_score(y_val, y_val_pred_downsampled)
downsampled_auc = roc_auc_score(y_val, downsampled_model.predict_proba(X_val)[:, 1])

print("\nDownsampling model results:")
print(f"F1 Score: {downsampled_f1:.4f}")
print(f"AUC-ROC: {downsampled_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_downsampled))

# In[20]

# Compare all models
models = {
    'Baseline': {
        'F1': baseline_f1,
        'AUC': baseline_auc
    },
    'Class Weights': {
        'F1': weighted_f1,
        'AUC': weighted_auc
    },
    'Upsampling': {
        'F1': upsampled_f1,
        'AUC': upsampled_auc
    },
    'Downsampling': {
        'F1': downsampled_f1,
        'AUC': downsampled_auc
    }
}

# Create a DataFrame for easy comparison
comparison_df = pd.DataFrame(models).T
comparison_df = comparison_df.sort_values('F1', ascending=False)
print("\nModel Comparison:")
print(comparison_df)

# Select the best model based on F1 score
best_model_name = comparison_df.index[0]
print(f"\nBest model based on F1 score: {best_model_name}")

# Select the appropriate model for hyperparameter tuning
if best_model_name == 'Class Weights':
    best_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    X_train_best, y_train_best = X_train, y_train
elif best_model_name == 'Upsampling':
    best_model = RandomForestClassifier(random_state=42)
    X_train_best, y_train_best = X_train_upsampled, y_train_upsampled
elif best_model_name == 'Downsampling':
    best_model = RandomForestClassifier(random_state=42)
    X_train_best, y_train_best = X_train_downsampled, y_train_downsampled
else:
    best_model = RandomForestClassifier(random_state=42)
    X_train_best, y_train_best = X_train, y_train

# In[21]
# Step 5.1: Executive Summary of Model Approaches

print("="*60)
print("EXECUTIVE SUMMARY - MODEL APPROACH COMPARISON")
print("="*60)

print(f"""
BUSINESS CONTEXT:
Our dataset shows 80% customer retention vs 20% churn, creating a prediction    challenge where standard models favor the majority class and poorly identify at-risk customers.

APPROACH RESULTS:
- Baseline Model: F1 Score {baseline_f1:.4f} - Inadequate churn detection due to class imbalance
- Class Weights: F1 Score {weighted_f1:.4f} - Improved by penalizing misclassification of churners  
- Upsampling: F1 Score {upsampled_f1:.4f} - Enhanced by creating synthetic churn examples for training
- Downsampling: F1 Score {downsampled_f1:.4f} - Balanced by reducing non-churn training samples

RECOMMENDATION:
The {best_model_name} approach achieved our highest F1 score of {comparison_df.iloc[0]['F1']:.4f}, 
providing optimal churn identification for proactive customer retention strategies.
""")

print("="*60)

# In[23]
# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4]
}

print("HYPERPARAMETER TUNING FOR ALL MODELS")

# Dictionary to store tuned models and their performance
tuned_models_results = {}

# 1. Tune Baseline Model
print("\n1. Tuning Baseline Model...")
baseline_model = RandomForestClassifier(random_state=42)
baseline_grid = GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
baseline_grid.fit(X_train, y_train)

# Evaluate tuned baseline model
baseline_tuned = baseline_grid.best_estimator_
y_val_pred_baseline_tuned = baseline_tuned.predict(X_val)
baseline_tuned_f1 = f1_score(y_val, y_val_pred_baseline_tuned)
baseline_tuned_auc = roc_auc_score(y_val, baseline_tuned.predict_proba(X_val)[:, 1])

tuned_models_results['Baseline'] = {
    'model': baseline_tuned,
    'best_params': baseline_grid.best_params_,
    'cv_f1': baseline_grid.best_score_,
    'val_f1': baseline_tuned_f1,
    'val_auc': baseline_tuned_auc,
    'training_data': (X_train, y_train)
}

print(f"Best parameters: {baseline_grid.best_params_}")
print(f"Best CV F1 score: {baseline_grid.best_score_:.4f}")
print(f"Validation F1 score: {baseline_tuned_f1:.4f}")

# In[24]
# 2. Tune Class Weights Model
print("\n2. Tuning Class Weights Model...")
# Modify param_grid to include class_weight
param_grid_weighted = param_grid.copy()
param_grid_weighted['class_weight'] = ['balanced']

weighted_model = RandomForestClassifier(random_state=42)
weighted_grid = GridSearchCV(
    estimator=weighted_model,
    param_grid=param_grid_weighted,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
weighted_grid.fit(X_train, y_train)

# Evaluate tuned weighted model
weighted_tuned = weighted_grid.best_estimator_
y_val_pred_weighted_tuned = weighted_tuned.predict(X_val)
weighted_tuned_f1 = f1_score(y_val, y_val_pred_weighted_tuned)
weighted_tuned_auc = roc_auc_score(y_val, weighted_tuned.predict_proba(X_val)[:, 1])

tuned_models_results['Class Weights'] = {
    'model': weighted_tuned,
    'best_params': weighted_grid.best_params_,
    'cv_f1': weighted_grid.best_score_,
    'val_f1': weighted_tuned_f1,
    'val_auc': weighted_tuned_auc,
    'training_data': (X_train, y_train)
}

print(f"Best parameters: {weighted_grid.best_params_}")
print(f"Best CV F1 score: {weighted_grid.best_score_:.4f}")
print(f"Validation F1 score: {weighted_tuned_f1:.4f}")

# In[25]
# 3. Tune Upsampling Model
print("\n3. Tuning Upsampling Model...")
upsampling_model = RandomForestClassifier(random_state=42)
upsampling_grid = GridSearchCV(
    estimator=upsampling_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
upsampling_grid.fit(X_train_upsampled, y_train_upsampled)

# Evaluate tuned upsampling model
upsampling_tuned = upsampling_grid.best_estimator_
y_val_pred_upsampling_tuned = upsampling_tuned.predict(X_val)
upsampling_tuned_f1 = f1_score(y_val, y_val_pred_upsampling_tuned)
upsampling_tuned_auc = roc_auc_score(y_val, upsampling_tuned.predict_proba(X_val)[:, 1])

tuned_models_results['Upsampling'] = {
    'model': upsampling_tuned,
    'best_params': upsampling_grid.best_params_,
    'cv_f1': upsampling_grid.best_score_,
    'val_f1': upsampling_tuned_f1,
    'val_auc': upsampling_tuned_auc,
    'training_data': (X_train_upsampled, y_train_upsampled)
}

print(f"Best parameters: {upsampling_grid.best_params_}")
print(f"Best CV F1 score: {upsampling_grid.best_score_:.4f}")
print(f"Validation F1 score: {upsampling_tuned_f1:.4f}")

# In[26]
# 4. Tune Downsampling Model
print("\n4. Tuning Downsampling Model...")
downsampling_model = RandomForestClassifier(random_state=42)
downsampling_grid = GridSearchCV(
    estimator=downsampling_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
downsampling_grid.fit(X_train_downsampled, y_train_downsampled)

# Evaluate tuned downsampling model
downsampling_tuned = downsampling_grid.best_estimator_
y_val_pred_downsampling_tuned = downsampling_tuned.predict(X_val)
downsampling_tuned_f1 = f1_score(y_val, y_val_pred_downsampling_tuned)
downsampling_tuned_auc = roc_auc_score(y_val, downsampling_tuned.predict_proba(X_val)[:, 1])

tuned_models_results['Downsampling'] = {
    'model': downsampling_tuned,
    'best_params': downsampling_grid.best_params_,
    'cv_f1': downsampling_grid.best_score_,
    'val_f1': downsampling_tuned_f1,
    'val_auc': downsampling_tuned_auc,
    'training_data': (X_train_downsampled, y_train_downsampled)
}

print(f"Best parameters: {downsampling_grid.best_params_}")
print(f"Best CV F1 score: {downsampling_grid.best_score_:.4f}")
print(f"Validation F1 score: {downsampling_tuned_f1:.4f}")

# In[28]
# Create comparison DataFrame
comparison_data = []
for model_name, results in tuned_models_results.items():
    comparison_data.append({
        'Model': model_name,
        'CV_F1': results['cv_f1'],
        'Validation_F1': results['val_f1'],
        'Validation_AUC': results['val_auc']
    })

tuned_comparison_df = pd.DataFrame(comparison_data)
tuned_comparison_df = tuned_comparison_df.sort_values('Validation_F1', ascending=False)
print("\nTuned Models Performance:")
print(tuned_comparison_df.to_string(index=False, float_format='%.4f'))

# Select the best model based on validation F1 score
best_model_name = tuned_comparison_df.iloc[0]['Model']
best_model_info = tuned_models_results[best_model_name]
final_model = best_model_info['model']

print(f"\n BEST MODEL: {best_model_name}")
print(f"Validation F1 Score: {best_model_info['val_f1']:.4f}")
print(f"Validation AUC Score: {best_model_info['val_auc']:.4f}")
print(f"Best Parameters: {best_model_info['best_params']}")

# In[30]
y_val_pred_final = final_model.predict(X_val)
final_f1 = f1_score(y_val, y_val_pred_final)
final_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])

print(f"\nFinal Model Performance on Validation Set:")
print(f"F1 Score: {final_f1:.4f}")
print(f"AUC-ROC: {final_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_final))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_final))

# In[33]
# Evaluate the final tuned model on the test set
y_test_pred = final_model.predict(X_test)
final_f1 = f1_score(y_test, y_test_pred)
final_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])


print("FINAL TEST RESULTS:")

print(f"F1 Score: {final_f1:.4f}")
print(f"AUC-ROC: {final_auc:.4f}")
print(f"Project Requirement: F1 >= 0.59")
print(f"Requirement Met: {'YES' if final_f1 >= 0.59 else 'NO'}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# In[35]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, final_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(12, 5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {final_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Customer Churn Prediction')
plt.legend()
plt.grid(True)

# Feature Importance
plt.subplot(1, 2, 2)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.barh(range(len(feature_importance)), feature_importance['Importance'])
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Customer Churn Prediction')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# In[37]

top_features = feature_importance.head(5)
print("\nKey Drivers of Customer Churn:")
for idx, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"{idx}. {row['Feature']}: {row['Importance']:.4f}")

