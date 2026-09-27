import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Setup aur Medical Disclaimer
st.set_page_config(page_title="Heart Risk Screener", layout="centered")
st.title("🫀 Heart Disease Risk Screener")
st.warning("**DISCLAIMER:** This tool is for educational purposes only and DOES NOT provide medical advice or diagnosis. Please consult a doctor for any health concerns.")
st.markdown("""
<style>
    /* 1. Premium Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
    }
    
    /* 2. Text aur Headings ka style */
    html, body, [class*="css"] {
        font-family: 'Trebuchet MS', 'Arial', sans-serif;
    }
    h1 {
        color: #d63031 !important;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #0984e3 !important;
    }

    /* 3. BUBBLE STYLE FOR BOXES */
    /* Input aur Select boxes ko gol (bubble) banana */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div {
        border-radius: 25px !important; /* Gol shape ke liye */
        border: 2px solid #74b9ff !important; /* Halka neela border */
        background-color: #ffffff !important; /* Andar se white */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08) !important; /* 3D Shadow */
        padding: 2px 10px !important;
        transition: all 0.3s ease;
    }

    /* Hover karne par box thoda glow karega */
    div[data-baseweb="input"] > div:hover, 
    div[data-baseweb="select"] > div:hover {
        border-color: #0984e3 !important;
        box-shadow: 0 6px 12px rgba(9, 132, 227, 0.2) !important;
    }

    /* Andar likhe hue numbers/text ka color dark aur bold karna */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] div {
        color: #2d3436 !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)
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
