import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Setup aur Medical Disclaimer
st.set_page_config(page_title="Heart Risk Screener", layout="centered")
st.title("🫀 Heart Disease Risk Screener")
st.warning("**DISCLAIMER:** This tool is for educational purposes only and DOES NOT provide medical advice or diagnosis. Please consult a doctor for any health concerns.")
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(270deg,
            rgb(255,0,0), rgb(255,165,0), rgb(255,255,0),
            rgb(0,255,0), rgb(0,255,255), rgb(0,0,255), rgb(148,0,211));
        background-size: 1400% 1400%;
        animation: bgflow 8s ease infinite;
    }
    @keyframes bgflow {
        0%   {background-position: 0% 50%;}
        50%  {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    h1, h2, h3 {
        animation: textcolor 15s linear infinite;
        text-shadow: 0 0 8px rgb(0,0,0);
        text-align: center;
    }
    @keyframes textcolor {
        0%   {color: rgb(255,0,0);}
        33%  {color: rgb(0,255,0);}
        66%  {color: rgb(0,120,255);}
        100% {color: rgb(255,0,0);}
    }

    .stButton>button {
        background: rgb(15,15,15);
        color: rgb(255,255,255);
        border-radius: 12px;
        border: 3px solid rgb(255,0,0);
        animation: glow 4s linear infinite;
    }
    @keyframes glow {
        0%   {border-color: rgb(255,0,0); box-shadow: 0 0 12px rgb(255,0,0);}
        33%  {border-color: rgb(0,255,0); box-shadow: 0 0 12px rgb(0,255,0);}
        66%  {border-color: rgb(0,0,255); box-shadow: 0 0 12px rgb(0,0,255);}
        100% {border-color: rgb(255,0,0); box-shadow: 0 0 12px rgb(255,0,0);}
    }

    label, p { color: rgb(255,255,255) !important; text-shadow: 1px 1px 3px rgb(0,0,0); }
          /* Bubble boxes (strong version) */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInputContainer"] {
        border-radius: 25px !important;
        overflow: hidden;
    }
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInputContainer"] {
        border: 2px solid rgb(116,185,255) !important;
        background-color: rgb(255,255,255) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div {
        color: rgb(45,52,54) !important;
        font-weight: bold !important;
        background: transparent !important;
    }
    div[data-testid="stNumberInputContainer"] button {
        border-radius: 50% !important;
    }
    /* Poora widget box ubhra hua card */
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.35), rgba(255,255,255,0.12)) !important;
        border: 2px solid rgba(255,255,255,0.7) !important;
        border-radius: 25px !important;
        padding: 14px 16px 16px 16px !important;
        margin-bottom: 14px !important;
        box-shadow: 8px 10px 20px rgba(0,0,0,0.4),
                    -4px -4px 12px rgba(255,255,255,0.5),
                    inset 0 2px 4px rgba(255,255,255,0.7) !important;
        backdrop-filter: blur(6px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stNumberInput"]:hover,
    div[data-testid="stSelectbox"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 12px 18px 30px rgba(0,0,0,0.5),
                    -4px -4px 12px rgba(255,255,255,0.6) !important;
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
