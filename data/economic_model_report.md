
# Economic Model Report

## Overview
- **Dataset**: Economic indicators for 20 countries from 2000-2023
- **Features**: GDP growth rate, inflation rate, and derived metrics
- **Target**: Economic status (booming, stable, or shrinking)
- **Model**: Random Forest Classifier

## Model Performance
              precision    recall  f1-score   support

     booming       0.97      0.95      0.96        37
   shrinking       1.00      1.00      1.00        16
      stable       0.98      0.99      0.98        91

    accuracy                           0.98       144
   macro avg       0.98      0.98      0.98       144
weighted avg       0.98      0.98      0.98       144


## Feature Importance
                  Feature  Importance
0         gdp_growth_rate    0.428174
1          inflation_rate    0.160428
2  growth_inflation_ratio    0.123197
3         economic_health    0.116453
6       gdp_growth_change    0.041065
7           gdp_3yr_trend    0.040020
5          prev_inflation    0.029962
4         prev_gdp_growth    0.029897
8           inf_3yr_trend    0.015563
9        growth_stability    0.015241

## Country Analysis
- Countries with highest percentage of booming years: China, Korea, Rep., India
- Countries with highest percentage of shrinking years: Italy, Mexico, Spain

## Methodology
1. **Data Preprocessing**: Removed missing values and scaled features
2. **Feature Engineering**: Created derived metrics including:
   - Growth-inflation ratio
   - Economic health (GDP growth - inflation)
   - Year-over-year changes in growth and inflation
   - 3-year trend analysis
   - Growth stability metrics
3. **Classification**: 
   - Booming: GDP growth ≥ 3% with inflation < 5%
   - Shrinking: GDP growth ≤ 0%
   - Stable: All other economies

## Visualizations
Generated visualizations for confusion matrix, feature importance, and country-specific analysis.
