# Economic Model Visualizer

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-green?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

## 📊 Overview

Economic Model Visualizer is an interactive web application for analyzing economic trends and predicting economic status using machine learning. The tool provides data visualization, predictive analytics, and forecasting capabilities for economic indicators.

## ✨ Features

- **Interactive Data Visualization** - Explore economic trends across multiple countries
- **ML-Powered Predictions** - Classify economic status using Random Forest models
- **Future Economic Outlook** - Project GDP growth and inflation trends
- **Country Comparison** - Analyze economic similarities between nations
- **Responsive Design** - Optimized for desktop and mobile devices

## 🤖 Model Overview

The core prediction model classifies economic status into three categories:

| Status | Definition | Criteria |
|--------|------------|----------|
| **Booming** | Strong growth with controlled inflation | GDP growth ≥ 3.0% AND inflation < 5.0% |
| **Shrinking** | Economic contraction | GDP growth ≤ 0% |
| **Stable** | Moderate growth | All other cases |

### Key Features in Prediction

- **Growth-Inflation Ratio**: Measures economic efficiency by comparing growth to inflation
- **Economic Health**: Captures real value creation accounting for inflation
- **3-Year Trends**: Analyzes growth and inflation trajectories over time

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/harshitkumar9030/analyzing_economies.git
cd alanyzing_economies

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 📊 Usage Examples

### Making Economic Predictions

1. Enter GDP growth and inflation values
2. Click "Predict Economic Status"
3. View prediction results and probability breakdown
4. Analyze similar historical economies

### Exploring Forecasts

1. Select a country and forecast horizon
2. Generate economic projections
3. Visualize future economic status timeline
4. Review confidence intervals

## 📝 Technical Documentation

For detailed information on the mathematical models and methodology, see the [Technical Documentation](http://localhost:5000/technical-docs) page in the application.

## 👥 Contributors

- Harshit - Lead Developer

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- World Bank Open Data Repository
- International Monetary Fund (IMF) Economic Outlook Reports
- Organisation for Economic Co-operation and Development (OECD) Data
