# Telecom User Analysis Project

## Overview
Analysis of telecom user data focusing on overview, engagement, experience, and satisfaction metrics using Python, SQL, and Streamlit.

## Features
- Handset usage and application behavior analysis
- Session metrics and user clustering
- Network parameter analysis 
- Satisfaction scoring and prediction
- Interactive Streamlit dashboard

## Setup
```bash
git clone <repository-url>
pip install -r requirements.txt
cp .env.example .env  # Configure your env variables
```

## Usage
```bash
# Start dashboard
python scripts/Dashboard.py

# Run analysis scripts
python scripts/User_Overview_Analysis.py
python scripts/User_Engagement_Analysis.py
```

## Docker Setup
```bash
docker build -t telecom-analysis .
docker run -p 8501:8501 telecom-analysis
```

## Testing
```bash
python -m pytest tests/
```

## Project Structure
```
├── .vscode/
├── .github/workflows/
│   └── unittests.yml
├── notebooks/
├── scripts/
├── src/
├── tests/
└── requirements.txt
```



# scripts/README.md
# Analysis Scripts

Production-ready analysis pipeline implementations.

## Scripts
- `Dashboard.py`: Streamlit interface
- `User_Overview_Analysis.py`: Overview metrics
- `User_Engagement_Analysis.py`: Engagement analysis
- `UserExperience.py`: Experience calculations
- `User_Satisfaction.py`: Satisfaction modeling



# .env.example
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=password
DB_NAME=telecom_analysis

# requirements.txt
streamlit==1.28.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.1
plotly==5.17.0
mysql-connector-python==8.1.0
python-dotenv==1.0.0
pytest==7.4.2