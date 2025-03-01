import pandas as pd
import pandas_datareader.wb as wb
import os

# Define the same list of countries as in data.py for consistency
countries = ['US', 'CA', 'MX', 'GB', 'DE', 'FR', 'JP', 'CN', 'IN', 'BR', 'RU', 
             'AU', 'IT', 'KR', 'ES', 'ID', 'SA', 'TR', 'CH', 'NL']

# Fetch inflation data (Consumer Price Index) for selected countries
inflation_data = wb.download(indicator='FP.CPI.TOTL.ZG', 
                           country=countries,
                           start=2000, 
                           end=2024)

# Reset index to make country and year columns accessible
inflation_data = inflation_data.reset_index()

print("Inflation Data Preview:")
print(inflation_data.head())

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

# Save the inflation data to CSV file
inflation_file_path = os.path.join(data_dir, 'inflation_data.csv')
inflation_data.to_csv(inflation_file_path, index=False)
print(f"Inflation data saved to {inflation_file_path}")

# Load the existing GDP growth data
gdp_file_path = os.path.join(data_dir, 'gdp_growth_data.csv')
gdp_data = pd.read_csv(gdp_file_path)

# Rename the indicator columns for clarity before merging
inflation_data = inflation_data.rename(columns={'FP.CPI.TOTL.ZG': 'inflation_rate'})
gdp_data = gdp_data.rename(columns={'NY.GDP.MKTP.KD.ZG': 'gdp_growth_rate'})

# Convert 'year' to the same data type in both DataFrames
inflation_data['year'] = inflation_data['year'].astype(int)
gdp_data['year'] = gdp_data['year'].astype(int)

# Merge the GDP growth and inflation datasets on country and year
combined_data = pd.merge(gdp_data, inflation_data, on=['country', 'year'], how='outer')

# Save the combined dataset
combined_file_path = os.path.join(data_dir, 'economic_indicators.csv')
combined_data.to_csv(combined_file_path, index=False)
print(f"Combined economic data saved to {combined_file_path}")

# Print statistics about the combined dataset
print("\nCombined Dataset Statistics:")
print(f"Total records: {len(combined_data)}")
print(f"Countries included: {combined_data['country'].nunique()}")
print(f"Year range: {combined_data['year'].min()} - {combined_data['year'].max()}")