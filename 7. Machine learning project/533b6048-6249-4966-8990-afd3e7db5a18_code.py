# In[2]
# Import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# In[3]
# Set random seed for reproducibility
np.random.seed(42)

# In[4]
print("Step 1: Loading and exploring the dataset...")
df = pd.read_csv('/datasets/users_behavior.csv')

# Display basic information about the dataset
print("\nDataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())

# In[5]
# Check for missing values

print("\nMissing values in each column:")
print(df.isnull().sum())

# In[6]
# Check class distribution

print("\nClass distribution:")
print(df['is_ultra'].value_counts())
print(df['is_ultra'].value_counts(normalize=True))

# In[7]
# Visualize the data

plt.figure(figsize=(16, 12))

# Histogram for each feature

features = ['calls', 'minutes', 'messages', 'mb_used']
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x=feature, hue='is_ultra', bins=30, kde=True)
    plt.title(f'Distribution of {feature} by plan type')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig('feature_distributions.png')

# In[8]
# Pairplot to visualize relationships between features

sns.pairplot(df, hue='is_ultra', vars=features)
plt.savefig('pairplot.png')

# In[9]
# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')

print("\nData exploration completed. Visualizations saved.")

# In[15]
# Split features and target
X = df.drop('is_ultra', axis=1)
y = df['is_ultra']

# First split: training+validation (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Second split: training (60% of total) and validation (20% of total)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# In[16]
# Check class distribution in each set
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nClass distribution in validation set:")
print(y_val.value_counts(normalize=True))
print("\nClass distribution in test set:")
print(y_test.value_counts(normalize=True))

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# In[18]
# Helper function to evaluate models
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"{model_name} - Validation Accuracy: {accuracy:.4f}")
    return accuracy, model

# 3.1 Logistic Regression with different C values
print("\nTraining Logistic Regression models:")
best_lr_accuracy = 0
best_lr_model = None
best_lr_c = 0

for c in [0.01, 0.1, 1, 10, 100]:
    lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
    accuracy, model = evaluate_model(lr, X_train_scaled, y_train, X_val_scaled, y_val, f"Logistic Regression (C={c})")
    
    if accuracy > best_lr_accuracy:
        best_lr_accuracy = accuracy
        best_lr_model = model
        best_lr_c = c

print(f"\nBest Logistic Regression model: C={best_lr_c}, Validation Accuracy: {best_lr_accuracy:.4f}")

# In[19]
# 3.2 Decision Tree with different max_depths
print("\nTraining Decision Tree models:")
best_dt_accuracy = 0
best_dt_model = None
best_dt_depth = 0

for depth in [3, 5, 7, 10, 15, 20, 25]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    accuracy, model = evaluate_model(dt, X_train, y_train, X_val, y_val, f"Decision Tree (max_depth={depth})")
    
    if accuracy > best_dt_accuracy:
        best_dt_accuracy = accuracy
        best_dt_model = model
        best_dt_depth = depth

print(f"\nBest Decision Tree model: max_depth={best_dt_depth}, Validation Accuracy: {best_dt_accuracy:.4f}")

# In[20]
# 3.3 Random Forest with different combinations of max_depth and n_estimators
print("\nTraining Random Forest models:")
best_rf_accuracy = 0
best_rf_model = None
best_rf_depth = 0
best_rf_n_estimators = 0

for depth in [5, 10, 15, 20, 25]:
    for n_estimators in [10, 16, 20, 50, 100]:
        rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, random_state=42)
        accuracy, model = evaluate_model(rf, X_train, y_train, X_val, y_val, f"Random Forest (max_depth={depth}, n_estimators={n_estimators})")
        
        if accuracy > best_rf_accuracy:
            best_rf_accuracy = accuracy
            best_rf_model = model
            best_rf_depth = depth
            best_rf_n_estimators = n_estimators

print(f"\nBest Random Forest model: max_depth={best_rf_depth}, n_estimators={best_rf_n_estimators}, Validation Accuracy: {best_rf_accuracy:.4f}")

# In[21]
# Find the best model overall
models = {
    "Logistic Regression": (best_lr_model, best_lr_accuracy),
    "Decision Tree": (best_dt_model, best_dt_accuracy),
    "Random Forest": (best_rf_model, best_rf_accuracy)
}

best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
best_model, best_val_accuracy = models[best_model_name]

print(f"\nBest model overall: {best_model_name}, Validation Accuracy: {best_val_accuracy:.4f}")

# In[24]
# Evaluate the best model on the test set
print("\nStep 4: Evaluating the best model on the test set...")

if best_model_name == "Logistic Regression":
    X_test_final = X_test_scaled
else:
    X_test_final = X_test

y_test_pred = best_model.predict(X_test_final)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nBest model ({best_model_name}) - Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# In[25]
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Smart (0)', 'Ultra (1)'])
plt.yticks([0.5, 1.5], ['Smart (0)', 'Ultra (1)'])
plt.savefig('confusion_matrix.png')

# In[27]
# Check if the model meets the threshold
if test_accuracy >= 0.75:
    print(f"\nThe model meets the accuracy threshold (>= 0.75) with a test accuracy of {test_accuracy:.4f}")
else:
    print(f"\nThe model does not meet the accuracy threshold (>= 0.75) with a test accuracy of {test_accuracy:.4f}")

# In[29]
# Create a balanced dataset with equal representation of both classes
smart_samples = df[df['is_ultra'] == 0].sample(n=min(df['is_ultra'].value_counts()), random_state=42)
ultra_samples = df[df['is_ultra'] == 1].sample(n=min(df['is_ultra'].value_counts()), random_state=42)
balanced_df = pd.concat([smart_samples, ultra_samples], axis=0)


# Split the balanced dataset
X_balanced = balanced_df.drop('is_ultra', axis=1)
y_balanced = balanced_df['is_ultra']

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)



# Train the best model on the balanced dataset
if best_model_name == "Logistic Regression":
    balanced_model = LogisticRegression(C=best_lr_c, max_iter=1000, random_state=42)
    X_train_bal_scaled = scaler.fit_transform(X_train_bal)
    X_test_bal_scaled = scaler.transform(X_test_bal)
    balanced_model.fit(X_train_bal_scaled, y_train_bal)
    y_test_bal_pred = balanced_model.predict(X_test_bal_scaled)
elif best_model_name == "Decision Tree":
    balanced_model = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
    balanced_model.fit(X_train_bal, y_train_bal)
    y_test_bal_pred = balanced_model.predict(X_test_bal)
else:  # Random Forest
    balanced_model = RandomForestClassifier(max_depth=best_rf_depth, n_estimators=best_rf_n_estimators, random_state=42)
    balanced_model.fit(X_train_bal, y_train_bal)
    y_test_bal_pred = balanced_model.predict(X_test_bal)

balanced_accuracy = accuracy_score(y_test_bal, y_test_bal_pred)

print(f"\nBest model ({best_model_name}) on balanced dataset - Test Accuracy: {balanced_accuracy:.4f}")
print("\nClassification Report (Balanced dataset):")
print(classification_report(y_test_bal, y_test_bal_pred))

# In[30]
# Compare with original accuracy
print(f"\nAccuracy on imbalanced dataset: {test_accuracy:.4f}")
print(f"Accuracy on balanced dataset: {balanced_accuracy:.4f}")
print(f"Difference: {test_accuracy - balanced_accuracy:.4f}")

# In[31]
# Feature importance analysis (for Tree-based models)
if best_model_name in ["Decision Tree", "Random Forest"]:
    print("\nFeature Importance:")
    importances = best_model.feature_importances_
    feature_names = X.columns
    
# Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("Feature ranking:")
    for i in range(X.shape[1]):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

# In[32]
# Summary of findings
print("\nSummary of Findings:")
print(f"1. The best model is {best_model_name} with test accuracy of {test_accuracy:.4f}")
print(f"2. The accuracy threshold of 0.75 {'is met' if test_accuracy >= 0.75 else 'is not met'}")
print(f"3. On a balanced dataset, the model achieved an accuracy of {balanced_accuracy:.4f}")
if best_model_name in ["Decision Tree", "Random Forest"]:
    print(f"4. The most important feature is {feature_names[indices[0]]}")

