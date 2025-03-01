
# Economic Model Report

## Overview
- **Dataset**: Economic indicators for 20 countries from 2000-2023
- **Features**: GDP growth rate, inflation rate, and derived metrics
- **Target**: Economic status (booming, stable, or shrinking)
- **Model**: Random Forest Classifier

## Model Performance
              precision    recall  f1-score   support

     booming       1.00      0.97      0.98        33
     shrinking       1.00      1.00      1.00        23
     stable       0.99      1.00      0.99        82

     accuracy                           0.99       138
     macro avg       1.00      0.99      0.99       138
     weighted avg       0.99      0.99      0.99       138


## Feature Importance
                  Feature  Importance
         0         gdp_growth_rate    0.432944
         3  growth_inflation_ratio    0.146464
         2         economic_health    0.143360
         1          inflation_rate    0.128742
         6       gdp_growth_change    0.063337
         5          prev_inflation    0.043406
         4         prev_gdp_growth    0.041748

## Country Analysis
- Countries with highest percentage of booming years: China, Korea, Rep., India
- Countries with highest percentage of shrinking years: Italy, Mexico, Spain

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
