from flask import Flask, render_template, jsonify
import pandas as pd
import os
import json
import pickle
import numpy as np

app = Flask(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None  # Convert NaN to null
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app.json_encoder = NpEncoder

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
        print(f"Warning: Model file not found at {model_path}")
        model = None
except FileNotFoundError as e:
    print(f"Error: {e}")
    data = pd.DataFrame({
        'country': ['US', 'CN', 'JP'],
        'year': [2020, 2020, 2020],
        'gdp_growth_rate': [2.1, 6.5, 1.2],
        'inflation_rate': [1.5, 2.5, 0.5]
    })
    data['economic_status'] = data.apply(classify_economy, axis=1)
    print("Created minimal test dataset")
    model = None

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

if __name__ == '__main__':
    app.run(debug=True)