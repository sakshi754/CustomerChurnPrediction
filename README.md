# Customer Churn Prediction
Run live app here https://customerchurnprediction-9urblju6cbsmz3funcuqjb.streamlit.app/ 
## Overview
This project is a **Customer Churn Prediction** system that helps businesses identify customers who are likely to churn (leave). It uses **machine learning** techniques to analyze customer data and predict churn probability. The project is implemented in **Python** and supports both exploratory analysis and model training.

## Features
- Loads and preprocesses **customer churn dataset**
- Trains multiple machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- Evaluates models using **Accuracy, Precision-Recall, and ROC-AUC**
- Provides feature importance insights using **SHAP**
- Supports model deployment using **Streamlit** (optional)

## Tech Stack
- **Python** (pandas, numpy, scikit-learn, xgboost, shap, joblib, matplotlib)
- **Machine Learning Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Feature Importance Analysis:** SHAP (SHapley Additive exPlanations)
- **Deployment:** Streamlit (Optional)

## Installation
To set up and run the project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CustomerChurnPrediction.git
cd CustomerChurnPrediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
Run the Python script to train models and evaluate churn prediction.
```bash
python churn_prediction.py
```

## Dataset
The project uses a **Customer Churn Dataset** that includes customer attributes such as:
- **Customer tenure**
- **Monthly charges**
- **Contract type**
- **Payment method**
- **Services subscribed (Internet, Phone, TV, etc.)**
- **Churn Label (Yes/No)**

## Project Workflow
1. **Load Data:** Load the dataset from a CSV file.
2. **Data Preprocessing:**
   - Convert categorical features into numerical values.
   - Handle missing values.
   - Normalize numerical data.
3. **Train Models:** Train multiple machine learning models to predict churn.
4. **Evaluate Performance:** Assess models using **ROC-AUC, Accuracy, and Precision-Recall**.
5. **Feature Importance Analysis:** Use **SHAP values** to interpret feature impact.
6. **Deploy Model (Optional):** Use **Streamlit** to create a simple web app for making predictions.

## How to Use
1. **Run the script** to train the models and evaluate churn predictions.
2. **For deployment (optional):**
   ```bash
   streamlit run churn_prediction.py
   ```
3. **Interact with the web app** (if Streamlit is enabled) to input customer details and get churn predictions.

## Results
- The **XGBoost model** achieves the highest accuracy and AUC score.
- **Feature Importance Analysis (SHAP):** The most significant features influencing churn include:
  - Contract Type
  - Tenure
  - Monthly Charges
  - Payment Method

## Future Improvements
- Implement **Deep Learning models (LSTMs)** for better churn prediction.
- Apply **Hyperparameter tuning** for optimized model performance.
- Extend deployment to **Flask/FastAPI** for production-ready applications.

## Author
-Sakshi Kiran Naik
- GitHub: https://github.com/sakshi754/CustomerChurnPrediction/

## License
This project is open-source and available under the MIT License.

