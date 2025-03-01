import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# Load the economic indicators data
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
data_path = os.path.join(data_dir, 'economic_indicators.csv')
data = pd.read_csv(data_path)

# Data preprocessing
data = data.dropna()  # Remove rows with missing values

# Feature engineering
data['growth_inflation_ratio'] = data['gdp_growth_rate'] / data['inflation_rate'].replace(0, 0.001)
data['economic_health'] = data['gdp_growth_rate'] - data['inflation_rate'] 

# Create country-year lag features
data = data.sort_values(['country', 'year'])
data['prev_gdp_growth'] = data.groupby('country')['gdp_growth_rate'].shift(1)
data['prev_inflation'] = data.groupby('country')['inflation_rate'].shift(1)
data['gdp_growth_change'] = data['gdp_growth_rate'] - data['prev_gdp_growth']
data = data.dropna()  # Remove rows with NaN from lag creation

# Define economic status (target variable)
def classify_economy(row):
    if row['gdp_growth_rate'] >= 3.0 and row['inflation_rate'] < 5.0:
        return 'booming'
    elif row['gdp_growth_rate'] <= 0:
        return 'shrinking'
    else:
        return 'stable'

data['economic_status'] = data.apply(classify_economy, axis=1)

# Prepare features and target
X = data[['gdp_growth_rate', 'inflation_rate', 'economic_health', 
          'growth_inflation_ratio', 'prev_gdp_growth', 'prev_inflation',
          'gdp_growth_change']]
y = data['economic_status']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred))

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['booming', 'shrinking', 'stable'],
            yticklabels=['booming', 'shrinking', 'stable'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(data_dir, 'confusion_matrix.png'))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Economic Status Prediction')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'feature_importance.png'))

# Generate country-specific insights
country_stats = data.groupby('country')['economic_status'].value_counts().unstack().fillna(0)
country_stats['total'] = country_stats.sum(axis=1)
country_stats['booming_pct'] = country_stats['booming'] / country_stats['total'] * 100
country_stats['shrinking_pct'] = country_stats['shrinking'] / country_stats['total'] * 100
country_stats = country_stats.sort_values('booming_pct', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='booming_pct', y=country_stats.index, data=country_stats.reset_index())
plt.title('Percentage of Years with Booming Economy by Country')
plt.xlabel('Percentage')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'country_booming_pct.png'))

# Save the model and report
import pickle
model_path = os.path.join(data_dir, 'economic_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

# Generate a summary report
report = f"""
# Economic Model Report

## Overview
- **Dataset**: Economic indicators for 20 countries from 2000-2023
- **Features**: GDP growth rate, inflation rate, and derived metrics
- **Target**: Economic status (booming, stable, or shrinking)
- **Model**: Random Forest Classifier

## Model Performance
{classification_report(y_test, y_pred)}

## Feature Importance
{feature_importance.to_string()}

## Country Analysis
- Countries with highest percentage of booming years: {', '.join(country_stats.head(3).index.tolist())}
- Countries with highest percentage of shrinking years: {', '.join(country_stats.sort_values('shrinking_pct', ascending=False).head(3).index.tolist())}

## Methodology
1. **Data Preprocessing**: Removed missing values and scaled features
2. **Feature Engineering**: Created derived metrics including:
   - Growth-inflation ratio
   - Economic health (GDP growth - inflation)
   - Year-over-year changes in growth and inflation
3. **Classification**: 
   - Booming: GDP growth ≥ 3% with inflation < 5%
   - Shrinking: GDP growth ≤ 0%
   - Stable: All other economies

## Visualizations
Generated visualizations for confusion matrix, feature importance, and country-specific analysis.
"""

report_path = os.path.join(data_dir, 'economic_model_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Report saved to {report_path}")