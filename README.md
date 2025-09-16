# Obesity Prediction Using Machine Learning

## Introduction
This project aims to predict obesity levels in individuals based on various health and lifestyle features using machine learning algorithms. Obesity is a growing health concern worldwide and early prediction can help in preventive measures and health management.

The dataset used for this project contains attributes such as age, height, weight, family history, physical activity, and dietary habits. The target variable, `NObeyesdad`, represents the obesity classification of individuals.

## Dataset
- **Source:** ObesityDataSet.csv  
- **Features:** 
  - Gender
  - Age
  - Height, Weight
  - Family history
  - Physical activity
  - Dietary habits
  - And others  
- **Target:** `NObeyesdad` (Obesity classification)  

## Project Workflow

### 1. Data Preprocessing
- Loaded the dataset using Pandas.
- Checked for missing values and basic statistics.
- Calculated BMI (`Weight / Height^2`) and added it as a feature.
- Encoded categorical variables using `LabelEncoder`.
- Split the dataset into features (`X`) and target (`y`).

### 2. Data Visualization
- Plotted the distribution of obesity classes using `countplot`.
- Created a correlation heatmap to analyze relationships between features.
- Examined the BMI distribution using a boxplot.

### 3. Feature Scaling
- Standardized features using `StandardScaler` to normalize the data for ML models.

### 4. Model Training and Evaluation
- Trained multiple classifiers:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Classifier
  - K-Nearest Neighbors
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Logged metrics and models to **MLflow** for experiment tracking.
- Selected the best model based on F1 Score.

### 5. ROC Curve
- Binarized the target variable for multi-class ROC curve analysis.
- Plotted ROC curves for each obesity class.
- Calculated AUC (Area Under Curve) for all classes.

### 6. Model Saving
- Saved the best model using `joblib`:
  - `best_obesity_model.pkl` → trained ML model
  - `scaler.pkl` → standard scaler for feature transformation
  - `label_encoders.pkl` → encoders for categorical features  

## Results
- The best performing model: **Random Forest / Gradient Boosting** (based on F1 Score)
- ROC curves show strong class-wise separation.
- Provides a reliable prediction pipeline for obesity classification.

## Conclusion
This project demonstrates a complete and robust machine learning workflow for obesity prediction. By combining data preprocessing, feature engineering, scaling, and multiple model evaluations, it identifies the best-performing classifier with high predictive accuracy. Visualizations such as correlation heatmaps and ROC curves enhance interpretability and provide insights into feature importance and model performance.  

The final saved model, along with preprocessing tools (scaler and label encoders), allows for efficient deployment and real-world application, helping healthcare professionals and individuals monitor obesity-related risks effectively. This project showcases how data-driven techniques can be leveraged for health monitoring and preventive care.

## Authors
- Varun.S
