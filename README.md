# Customer Churn Prediction Model

## Overview
This project implements a machine learning solution to predict customer churn using various classification algorithms. The model helps identify customers who are likely to discontinue services, enabling proactive retention strategies.

## Feature
- Data preprocessing and exploratory data analysis
- Implementation of multiple machine learning models:
  - XGBoost Classifier
  - Random Forest Classifier
  - Decision Tree Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Network
- Model evaluation and comparison
- Interactive web interface using Streamlit
- Model persistence using pickle
- Handling imbalanced data using SMOTE

## Installation

### Prerequisites
- Python 3.11
- Conda (recommended for environment management)

### Environment Setup

#### Create a new conda environment
```bash
conda create -n churn-prediction python=3.11
```

#### Activate the environment
```bash
conda activate churn-prediction
```

#### Install required packages
```bash
pip install -r requirements.txt
```

### Required Packages
- numpy
- pandas
- scikit-learn
- streamlit
- xgboost
- imbalanced-learn
- python-dotenv
- matplotlib
- seaborn
- plotly


## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook churn.ipynb
```

### Running the Streamlit App

```bash
streamlit run main.py
```


## Project Structure
customer-churn/
├── churn.ipynb # Main notebook with model development
├── main.py # Streamlit application
├── churn.csv # Dataset
├── models/ # Saved model files
│ ├── dt_model.pkl
│ ├── knn_model.pkl
│ ├── rf_model.pkl
│ ├── svm_model.pkl
│ └── xgb_model.pkl
├── requirements.txt # Project dependencies
└── README.md # Project documentation


## Model Performance
The project implements and compares several machine learning models for churn prediction. Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Environment Variables
Create a `.env` file in the project root with the following variables:
```
GROQ_API_KEY=<your-groq-api-key>
```


## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
