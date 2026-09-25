import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Setup aur Medical Disclaimer
st.set_page_config(page_title="Heart Risk Screener", layout="centered")
st.title("🫀 Heart Disease Risk Screener")
st.warning("**DISCLAIMER:** This tool is for educational purposes only and DOES NOT provide medical advice or diagnosis. Please consult a doctor for any health concerns.")

# 2. Data Load aur Model Train karna (Background mein)
@st.cache_data
def load_and_train():
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    # Wahi same model jo humne notebook mein use kiya tha
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, columns = load_and_train()

# 3. User se Input Lena
st.header("Enter Patient Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# 4. Result Button aur Prediction
if st.button("Calculate Risk"):
    # User ke data ka ek dataframe banana
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=columns)
    
    # Prediction aur Probability nikalna
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Screening Result:")
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected** (Probability: {probability:.1f}%)")
        st.write("This profile shows patterns associated with heart disease. Please consult a healthcare professional.")
    else:
        st.success(f"✅ **Low Risk Detected** (Probability: {probability:.1f}%)")
        st.write("This profile does not show strong patterns of heart disease. Maintain a healthy lifestyle!")