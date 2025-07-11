# American Express Fraud Detection System

A machine learning project focused on detecting fraudulent transactions using the American Express Kaggle dataset. This system compares the effectiveness of three different algorithms: XGBoost, CatBoost, and Logistic Regression.

## 📊 Project Overview

This project analyzes 5.5 million transaction records with 192 features to build predictive models for fraud detection. The study demonstrates how advanced machine learning algorithms can accurately identify fraudulent transactions, helping financial institutions prevent financial losses.

## 🎯 Key Results

- **CatBoost**: 98.1% accuracy (Best performing model)
- **XGBoost**: 97.7% accuracy  
- **Logistic Regression**: Baseline model for comparison

CatBoost achieved the lowest error rates with just 1.65% Type 1 error and 1.76% Type 2 error.

## 🔧 Technologies Used

- **Data Processing**: Dask, Pandas, NumPy
- **Machine Learning**: XGBoost, CatBoost, Scikit-learn
- **Data Visualization**: Plotly
- **Web Interface**: Streamlit
- **Data Balancing**: SMOTE (Synthetic Minority Oversampling Technique)

## 📋 Dataset Features

The dataset contains five categories of variables:
- **Delinquency variables** (D_*): 145 features
- **Spend variables** (S_*): Transaction spending patterns
- **Payment variables** (P_*): 3 features
- **Balance variables** (B_*): 42 features  
- **Risk variables** (R_*): Risk assessment metrics

## 🚀 Getting Started

### Prerequisites
```bash
pip install dask pandas numpy scikit-learn xgboost catboost plotly streamlit imbalanced-learn
```

### Data Preprocessing
1. **Missing Value Handling**: Columns with >75% missing values were dropped; remaining missing values imputed with mean
2. **Data Compression**: Dataset compressed from 55GB to 1.8GB using optimized data types
3. **Categorical Encoding**: Categorical variables converted to numerical format
4. **Class Balancing**: SMOTE applied to address the 75%/25% class imbalance

### Model Training
The project implements three classification models:
- Logistic Regression (baseline)
- XGBoost Classifier
- CatBoost Classifier

All models are saved in JSON format for deployment.

## 📈 Performance Metrics

| Model | Accuracy | Type 1 Error | Type 2 Error |
|-------|----------|--------------|--------------|
| Logistic Regression | - | 9.55% | 9.68% |
| XGBoost | 97.7% | 2.01% | 2.05% |
| CatBoost | 98.1% | 1.65% | 1.76% |

## 🖥️ Web Interface

The project includes a Streamlit-based web application that provides:
- Interactive data visualization
- Real-time fraud detection
- Model performance comparison
- Exploratory data analysis dashboard

## 📚 References

1. Raj, S. Benson Edwin, and A. Annie Portia. "Analysis on credit card fraud detection methods." *International Conference on Computer, Communication and Electrical Technology (ICCCET)*, 2011.

2. Ghosh, S. and Reilly, D.L. "Credit card fraud detection with a neural network." *Twenty-Seventh Hawaii International Conference on System Sciences*, 1994.

3. George, Ms. Shelly Shiju, and Maneeksha C. Ashok. "Fraud Detection in Credit Card using Machine Learning." *National Conference on Emerging Computer Applications*, 2022.

---

*This project demonstrates the practical application of machine learning in financial fraud detection, highlighting CatBoost as the superior algorithm for this domain.*
