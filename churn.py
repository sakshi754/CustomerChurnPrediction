import pandas as pd
import numpy as np
import shap
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ---------------------------- STEP 1: Load Data ----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/Telecom_customer_churn.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ---------------------------- STEP 2: Preprocessing ----------------------------
st.title("Customer Churn Prediction")

# Drop customerID column
df.drop(columns=['customerID'], inplace=True, errors='ignore')

# Convert TotalCharges to numeric (fixes conversion errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values - apply median only to numeric columns
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)


# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------- STEP 3: Train Models ----------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")

# ---------------------------- STEP 4: Evaluate Models ----------------------------
st.subheader("Model Evaluation")
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.write(f"**{name}** - Accuracy: {acc:.2f}, ROC-AUC: {roc_auc:.2f}")

# ---------------------------- STEP 5: SHAP Feature Importance ----------------------------
st.subheader("Feature Importance using SHAP")
shap_explainer = shap.TreeExplainer(trained_models["XGBoost"])
shap_values = shap_explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
st.pyplot(plt)

# ---------------------------- STEP 6: Deploy Simple Prediction UI ----------------------------
st.subheader("Make a Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

user_df = pd.DataFrame([user_input])
user_df = scaler.transform(user_df)

if st.button("Predict Churn"):
    prediction = trained_models["XGBoost"].predict(user_df)[0]
    result = "Churn" if prediction == 1 else "No Churn"
    st.write(f"**Prediction: {result}**")
