from pandas_datareader import wb

# Fetch inflation data for all countries
inflation_data = wb.download(indicator='FP.CPI.TOTL.ZG', country='all', start=2020, end=2023)