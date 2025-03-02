import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Custom JSON encoder to handle NaN values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app.json_encoder = NpEncoder

# Fix data path for both local and Heroku environments
if 'DYNO' in os.environ:
    # We're on Heroku - need to locate data in root directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    data_path = os.path.join(data_dir, 'economic_indicators.csv')
    model_path = os.path.join(data_dir, 'economic_model.pkl')
else:
    # Local development
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    data_path = os.path.join(data_dir, 'economic_indicators.csv')
    model_path = os.path.join(data_dir, 'economic_model.pkl')

print(f"Looking for data at: {data_path}")
print(f"Looking for model at: {model_path}")

def classify_economy(row):
    if row['gdp_growth_rate'] >= 3.0 and row['inflation_rate'] < 5.0:
        return 'booming'
    elif row['gdp_growth_rate'] <= 0:
        return 'shrinking'
    else:
        return 'stable'

def create_sample_data():
    """Generate comprehensive sample economic data"""
    print("Creating comprehensive sample dataset")
    countries = ['US', 'CN', 'JP', 'DE', 'UK', 'FR', 'IT', 'CA', 'KR', 'AU', 
                'IN', 'BR', 'RU', 'MX', 'ID', 'TR', 'SA', 'CH', 'NL', 'ES']
    
    # Base economic characteristics by country
    base_gdp = {
        'US': 2.1, 'CN': 6.5, 'JP': 1.2, 'DE': 1.8, 'UK': 1.5, 
        'FR': 1.3, 'IT': 0.8, 'CA': 2.0, 'KR': 3.2, 'AU': 2.5,
        'IN': 5.5, 'BR': 1.9, 'RU': 1.4, 'MX': 0.1, 'ID': 4.8,
        'TR': 2.8, 'SA': 0.7, 'CH': 1.1, 'NL': 1.6, 'ES': 1.0
    }
    
    base_inflation = {
        'US': 1.5, 'CN': 2.5, 'JP': 0.5, 'DE': 1.2, 'UK': 1.8, 
        'FR': 1.1, 'IT': 0.9, 'CA': 1.7, 'KR': 1.5, 'AU': 1.9,
        'IN': 3.5, 'BR': 3.8, 'RU': 4.2, 'MX': 3.0, 'ID': 2.8,
        'TR': 7.5, 'SA': 1.2, 'CH': 0.3, 'NL': 1.4, 'ES': 0.8
    }
    
    # Economic era modifiers
    eras = {
        # Pre-financial crisis
        (2000, 2007): {'gdp': 1.2, 'inf': 0.8},
        # Financial crisis
        (2008, 2010): {'gdp': 0.4, 'inf': 1.2},
        # Post-crisis recovery
        (2011, 2019): {'gdp': 1.0, 'inf': 0.9},
        # Covid pandemic
        (2020, 2021): {'gdp': 0.3, 'inf': 1.3},
        # Post-covid
        (2022, 2023): {'gdp': 0.9, 'inf': 1.8}
    }
    
    all_data = []
    for year in range(2000, 2024):
        # Find applicable economic era
        era_mod = next((mod for (start, end), mod in eras.items() if start <= year <= end), {'gdp': 1.0, 'inf': 1.0})
        
        for country in countries:
            gdp_growth = base_gdp[country] * era_mod['gdp'] * (0.7 + np.random.random() * 0.6)
            inflation = base_inflation[country] * era_mod['inf'] * (0.8 + np.random.random() * 0.4)
            
            all_data.append({
                'country': country,
                'year': year,
                'gdp_growth_rate': gdp_growth,
                'inflation_rate': inflation
            })
    
    df = pd.DataFrame(all_data)
    df['economic_status'] = df.apply(classify_economy, axis=1)
    
    print(f"Created sample dataset with {len(df)} rows for {len(countries)} countries")
    return df

def add_features(df):
    """Add engineered features to the dataframe"""
    # Safety check to prevent division by zero
    df['growth_inflation_ratio'] = df['gdp_growth_rate'] / df['inflation_rate'].replace(0, 0.001)
    df['economic_health'] = df['gdp_growth_rate'] - df['inflation_rate']
    
    # Add trend features
    countries = df['country'].unique()
    years = sorted(df['year'].unique())
    
    # Initialize new columns
    df['gdp_3yr_trend'] = 0.0
    df['inf_3yr_trend'] = 0.0
    df['growth_stability'] = 0.0
    
    for country in countries:
        country_data = df[df['country'] == country].sort_values('year')
        
        # Calculate 3-year trends and stability
        for i, year in enumerate(years):
            if i >= 2:  # Need at least 3 years of data
                past_3yrs = country_data[country_data['year'].isin(years[i-2:i+1])]
                
                if len(past_3yrs) == 3:
                    # GDP 3-year trend (positive = improving, negative = deteriorating)
                    gdp_vals = past_3yrs['gdp_growth_rate'].values
                    gdp_trend = gdp_vals[2] - gdp_vals[0]  # Latest minus oldest
                    
                    # Inflation 3-year trend
                    inf_vals = past_3yrs['inflation_rate'].values
                    inf_trend = inf_vals[2] - inf_vals[0]
                    
                    # Growth stability (lower = more stable)
                    stability = gdp_vals.std()
                    
                    # Update the values
                    idx = df[(df['country'] == country) & (df['year'] == year)].index
                    if not idx.empty:
                        df.loc[idx, 'gdp_3yr_trend'] = gdp_trend
                        df.loc[idx, 'inf_3yr_trend'] = inf_trend
                        df.loc[idx, 'growth_stability'] = stability
    
    return df

def check_model_features():
    """Check if the model's features match our expected feature set"""
    try:
        expected_features = ['gdp_growth_rate', 'inflation_rate', 'growth_inflation_ratio', 
                           'economic_health', 'gdp_3yr_trend', 'inf_3yr_trend', 'growth_stability']
        
        # For older scikit-learn versions
        if not hasattr(model, 'feature_names_in_'):
            print("Using an older scikit-learn version without feature_names_in_ attribute.")
            # We'll still check that the model was trained with the right number of features
            return True
        
        model_features = list(model.feature_names_in_)
        print(f"Model features: {model_features}")
        
        # Check if we need to retrain the model
        if len(model_features) != len(expected_features):
            print(f"Feature count mismatch! Model has {len(model_features)} features, but we expect {len(expected_features)}")
            return False
        
        return True
    except Exception as e:
        print(f"Error checking model features: {e}")
        return False

# Load or create data
try:
    data = pd.read_csv(data_path)
    print(f"Successfully loaded data with {len(data)} rows")
    
    if 'economic_status' not in data.columns:
        print("Adding economic_status classification to data")
        data['economic_status'] = data.apply(classify_economy, axis=1)
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Successfully loaded model")
    else:
        print("Model not found, training a new one...")
        # Feature engineering
        data = add_features(data)
        
        # Get feature columns in a predictable order
        feature_columns = ['gdp_growth_rate', 'inflation_rate', 'growth_inflation_ratio', 
                           'economic_health', 'gdp_3yr_trend', 'inf_3yr_trend', 'growth_stability']

        # Create the model ensuring we use the same feature order
        X = data[feature_columns]  # Use this instead of listing them inline
        y = data['economic_status']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        check_model_features()  # This will print debug info about the model features
        print("Model training complete")
except FileNotFoundError as e:
    print(f"Error: {e}")
    data = create_sample_data()
    
    # Feature engineering for sample data
    data = add_features(data)
    
    # Get feature columns in a predictable order
    feature_columns = ['gdp_growth_rate', 'inflation_rate', 'growth_inflation_ratio', 
                       'economic_health', 'gdp_3yr_trend', 'inf_3yr_trend', 'growth_stability']

    # Create and train model on the sample data
    X = data[feature_columns]  # Use this instead of listing them inline
    y = data['economic_status']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    check_model_features()  # This will print debug info about the model features
    print("Model trained on sample data")

# Create the scaler for standardizing inputs
scaler = StandardScaler()
scaler.fit(data[['gdp_growth_rate', 'inflation_rate']])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/countries')
def get_countries():
    countries = data['country'].unique().tolist()
    return jsonify(countries)

@app.route('/api/data')
def get_data():
    # Create pivot tables but then transpose them for better JavaScript access
    pivot_gdp = data.pivot(index='country', columns='year', values='gdp_growth_rate')
    pivot_inflation = data.pivot(index='country', columns='year', values='inflation_rate')
    pivot_status = data.pivot(index='country', columns='year', values='economic_status')
    
    result = {
        'countries': data['country'].unique().tolist(),
        'years': sorted(data['year'].unique().tolist()),
        'gdp_growth': pivot_gdp.replace({np.nan: None}).to_dict('index'),
        'inflation': pivot_inflation.replace({np.nan: None}).to_dict('index'),
        'economic_status': pivot_status.replace({np.nan: None}).to_dict('index')
    }
    return jsonify(result)

@app.route('/api/stats')
def get_stats():
    country_stats = data.groupby('country')['economic_status'].value_counts().unstack().fillna(0)
    country_stats['total'] = country_stats.sum(axis=1)
    
    for status in ['booming', 'stable', 'shrinking']:
        if status not in country_stats.columns:
            country_stats[status] = 0
    
    country_stats['booming_pct'] = (country_stats['booming'] / country_stats['total'] * 100).round(1)
    country_stats['shrinking_pct'] = (country_stats['shrinking'] / country_stats['total'] * 100).round(1)
    
    stats_dict = country_stats.replace({np.nan: None}).to_dict()
    return jsonify(stats_dict)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get values from request
        req_data = request.get_json()
        if req_data is None:
            return jsonify({'error': 'No JSON data received'}), 400
        
        print(f"Received prediction request: {req_data}")
        
        # Check for required fields
        if 'gdp_growth' not in req_data or 'inflation' not in req_data:
            return jsonify({'error': 'Missing required fields: gdp_growth and/or inflation'}), 400
            
        try:
            gdp_growth = float(req_data.get('gdp_growth', 0))
            inflation = float(req_data.get('inflation', 0))
        except (TypeError, ValueError) as e:
            return jsonify({'error': f'Invalid numeric values: {str(e)}'}), 400
        
        # Print debug information
        print(f"Input values: GDP growth = {gdp_growth}, Inflation = {inflation}")
        
        # Create prediction input with all required features
        gdp_growth_rate = gdp_growth
        inflation_rate = inflation
        
        # Avoid division by zero
        if inflation_rate == 0:
            growth_inflation_ratio = gdp_growth_rate / 0.001
        else:
            growth_inflation_ratio = gdp_growth_rate / inflation_rate
            
        economic_health = gdp_growth_rate - inflation_rate
        
        # For new data points, we don't have trend/stability data 
        # so use median values from our dataset
        gdp_3yr_trend = data['gdp_3yr_trend'].median() if 'gdp_3yr_trend' in data.columns else 0.0
        inf_3yr_trend = data['inf_3yr_trend'].median() if 'inf_3yr_trend' in data.columns else 0.0
        growth_stability = data['growth_stability'].median() if 'growth_stability' in data.columns else 1.0
            
        print(f"Calculated additional features: ratio={growth_inflation_ratio}, health={economic_health}")
        print(f"Using trends/stability: gdp trend={gdp_3yr_trend}, inf trend={inf_3yr_trend}, stability={growth_stability}")
        
        # Create input array with all 7 features
        input_data = np.array([[
            gdp_growth_rate, 
            inflation_rate, 
            growth_inflation_ratio,
            economic_health,
            gdp_3yr_trend,
            inf_3yr_trend,
            growth_stability
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        print(f"Prediction: {prediction}")
        
        # Calculate prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        class_indices = {c: i for i, c in enumerate(model.classes_)}
        class_probabilities = {
            'booming': round(float(probabilities[class_indices.get('booming', 0)]) * 100, 1) if 'booming' in class_indices else 0,
            'stable': round(float(probabilities[class_indices.get('stable', 0)]) * 100, 1) if 'stable' in class_indices else 0,
            'shrinking': round(float(probabilities[class_indices.get('shrinking', 0)]) * 100, 1) if 'shrinking' in class_indices else 0
        }
        
        # Find similar economies
        distances = []
        for _, row in data.iterrows():
            dist = np.sqrt((row['gdp_growth_rate'] - gdp_growth)**2 + (row['inflation_rate'] - inflation)**2)
            distances.append({
                'country': row['country'],
                'year': int(row['year']),
                'distance': float(dist),
                'gdp': float(row['gdp_growth_rate']),
                'inflation': float(row['inflation_rate']),
                'status': row['economic_status']
            })
        
        similar_economies = sorted(distances, key=lambda x: x['distance'])[:5]
        
        result = {
            'prediction': prediction,
            'probabilities': class_probabilities,
            'similar_economies': similar_economies
        }
        print(f"Sending prediction response: {result}")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict-country/<country>', methods=['GET'])
def predict_country(country):
    try:
        if country not in data['country'].unique():
            return jsonify({'error': f'Country {country} not found in dataset'}), 404
        
        # Get the most recent data for this country
        country_data = data[data['country'] == country].sort_values('year', ascending=False)
        if len(country_data) == 0:
            return jsonify({'error': f'No data available for {country}'}), 404
            
        latest_data = country_data.iloc[0]
        
        # Get the next year
        next_year = int(latest_data['year']) + 1
        
        # Extract raw features for prediction
        gdp_growth_rate = float(latest_data['gdp_growth_rate'])
        inflation_rate = float(latest_data['inflation_rate'])
        
        # Calculate derived features (don't try to access them directly)
        # Avoid division by zero
        if inflation_rate == 0:
            growth_inflation_ratio = gdp_growth_rate / 0.001
        else:
            growth_inflation_ratio = gdp_growth_rate / inflation_rate
            
        economic_health = gdp_growth_rate - inflation_rate
        
        # Get trend features - we need to check if these columns exist in the DataFrame
        gdp_3yr_trend = 0.0
        inf_3yr_trend = 0.0
        growth_stability = 1.0
        
        # Try to get these from the latest_data if they exist
        if 'gdp_3yr_trend' in latest_data:
            gdp_3yr_trend = float(latest_data['gdp_3yr_trend'])
        elif 'gdp_3yr_trend' in data.columns:
            gdp_3yr_trend = data['gdp_3yr_trend'].median()
            
        if 'inf_3yr_trend' in latest_data:
            inf_3yr_trend = float(latest_data['inf_3yr_trend'])
        elif 'inf_3yr_trend' in data.columns:
            inf_3yr_trend = data['inf_3yr_trend'].median()
            
        if 'growth_stability' in latest_data:
            growth_stability = float(latest_data['growth_stability'])
        elif 'growth_stability' in data.columns:
            growth_stability = data['growth_stability'].median()
        
        # Create input array
        input_features = np.array([
            [
                gdp_growth_rate,
                inflation_rate,
                growth_inflation_ratio,
                economic_health,
                gdp_3yr_trend,
                inf_3yr_trend,
                growth_stability
            ]
        ])
        
        # Predict economic status
        prediction = model.predict(input_features)[0]
        
        # Calculate prediction probabilities
        probabilities = model.predict_proba(input_features)[0]
        class_indices = {c: i for i, c in enumerate(model.classes_)}
        class_probabilities = {
            'booming': round(float(probabilities[class_indices.get('booming', 0)]) * 100, 1) if 'booming' in class_indices else 0,
            'stable': round(float(probabilities[class_indices.get('stable', 0)]) * 100, 1) if 'stable' in class_indices else 0,
            'shrinking': round(float(probabilities[class_indices.get('shrinking', 0)]) * 100, 1) if 'shrinking' in class_indices else 0
        }
        
        return jsonify({
            'country': country,
            'prediction_year': next_year,
            'current_year': int(latest_data['year']),
            'current_gdp_growth': gdp_growth_rate,
            'current_inflation': inflation_rate,
            'current_status': latest_data['economic_status'],
            'predicted_status': prediction,
            'probabilities': class_probabilities
        })
        
    except Exception as e:
        import traceback
        print(f"Error in country prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict-future/<country>', methods=['GET'])
def predict_future(country):
    try:
        years_ahead = int(request.args.get('years', 5))  # Default: predict 5 years ahead
        if years_ahead < 1 or years_ahead > 10:
            return jsonify({'error': 'Years parameter must be between 1 and 10'}), 400
            
        if country not in data['country'].unique():
            return jsonify({'error': f'Country {country} not found in dataset'}), 404
        
        # Get the most recent data for this country
        country_data = data[data['country'] == country].sort_values('year', ascending=False)
        if len(country_data) == 0:
            return jsonify({'error': f'No data available for {country}'}), 404
            
        latest_data = country_data.iloc[0]
        current_year = int(latest_data['year'])
        
        # Calculate average GDP growth and inflation over the last 5 years
        recent_data = data[(data['country'] == country) & (data['year'] >= current_year - 5)]
        avg_gdp_growth = recent_data['gdp_growth_rate'].mean()
        avg_inflation = recent_data['inflation_rate'].mean()
        
        # Calculate growth and inflation volatility
        gdp_volatility = recent_data['gdp_growth_rate'].std() * 0.5
        inf_volatility = recent_data['inflation_rate'].std() * 0.5
        
        # Initialize with current values
        current_gdp = float(latest_data['gdp_growth_rate'])
        current_inflation = float(latest_data['inflation_rate'])
        
        # Generate predictions for multiple years
        predictions = []
        for i in range(1, years_ahead + 1):
            # Apply some trend reversion to the mean
            next_gdp = current_gdp * 0.7 + avg_gdp_growth * 0.3 + np.random.normal(0, gdp_volatility)
            next_inflation = current_inflation * 0.7 + avg_inflation * 0.3 + np.random.normal(0, inf_volatility)
            
            # Create feature vector for this prediction year
            growth_inflation_ratio = next_gdp / next_inflation if next_inflation != 0 else next_gdp / 0.001
            economic_health = next_gdp - next_inflation
            
            # For trend features, use values from data
            gdp_3yr_trend = data['gdp_3yr_trend'].median() if 'gdp_3yr_trend' in data.columns else 0.0
            inf_3yr_trend = data['inf_3yr_trend'].median() if 'inf_3yr_trend' in data.columns else 0.0
            growth_stability = data['growth_stability'].median() if 'growth_stability' in data.columns else 1.0
            
            # Create input array with all features
            input_features = np.array([
                [
                    next_gdp,
                    next_inflation,
                    growth_inflation_ratio,
                    economic_health,
                    gdp_3yr_trend,
                    inf_3yr_trend,
                    growth_stability
                ]
            ])
            
            # Predict economic status
            status = model.predict(input_features)[0]
            
            # Calculate confidence
            confidence = 100 - (i * 10)  # Confidence decreases with time
            confidence = max(confidence, 50)  # Minimum 50% confidence
            
            # Add to predictions list
            predictions.append({
                'year': current_year + i,
                'gdp_growth': round(float(next_gdp), 2),
                'inflation': round(float(next_inflation), 2),
                'status': status,
                'confidence': confidence
            })
            
            # Update current values for next iteration (carry forward)
            current_gdp = next_gdp
            current_inflation = next_inflation
        
        return jsonify({
            'country': country,
            'current_year': current_year,
            'predictions': predictions
        })
        
    except Exception as e:
        import traceback
        print(f"Error in future prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/technical-docs')
def technical_docs():
    return render_template('technical_docs.html')

if __name__ == '__main__':
    # Use environment port if available (Heroku sets this)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)