import streamlit as st
import pandas as pd
import joblib
import os
import re
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

# ================= DATABASE =================
DB_NAME = "heart_app.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# ---------- USERS ----------
def user_exists(username):
    conn = get_connection() 
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def register_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, password)
    )
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM users WHERE username=? AND password=?",
        (username, password)
    )
    ok = cur.fetchone() is not None
    conn.close()
    return ok

# ---------- PREDICTIONS ----------
def save_prediction(username, prob, risk, rf):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
        (username, probability, risk, risk_factors, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (username, prob, risk, rf, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_user_predictions(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT probability, risk, risk_factors, created_at
        FROM predictions
        WHERE username=?
        ORDER BY created_at DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ================= OCR =================
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    OCR_AVAILABLE = False

def extract_text(file):
    if not OCR_AVAILABLE:
        return ""
    if file.type == "application/pdf":
        images = convert_from_bytes(file.read())
        return " ".join(pytesseract.image_to_string(img) for img in images)
    return pytesseract.image_to_string(Image.open(file))

def find(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_report(text):
    return {
        "age": find(r"Age[:\s]+(\d+)", text),
        "bp": find(r"Blood\s*Pressure[:\s]+(\d+)", text),
        "chol": find(r"Cholesterol[:\s]+(\d+)", text),
        "fbs": find(r"Fasting\s*Blood\s*Sugar[:\s]+(\d+)", text),
    }

# ================= CONFIG =================
st.set_page_config(
    page_title="Heart Disease Risk Prediction System",
    layout="wide"
)
st.markdown(
    """
    <style>
    .auth-container {
        max-width: 400px;
        margin: auto;
        padding: 2rem;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()
model = joblib.load("rf_model.pkl")

# ================= SESSION =================
if "logged" not in st.session_state:
    st.session_state.logged = False
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"

# ================= AUTH =================
if not st.session_state.logged:


    if "logged" not in st.session_state:
     st.session_state.logged = False
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"

if not st.session_state.logged:
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # LOGIN
        if st.session_state.auth_page == "login":
            st.title("System Login")
            st.markdown("### Clinical Access Portal")
            
            with st.form("login_form"):
                u = st.text_input("Username", placeholder="admin")
                p = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                
                if submitted:
                    if authenticate_user(u, p):
                        st.session_state.logged = True
                        st.session_state.user = u
                        st.rerun()
                    else:
                        st.error("Invalid Credentials")

            st.write("---")
            if st.button("Create Account"):
                st.session_state.auth_page = "register"
                st.rerun()

        # REGISTER
        else: 
            st.title("New User Registration")
            with st.form("reg_form"):
                new_u = st.text_input("Desired Username")
                new_p = st.text_input("Desired Password", type="password")
                reg_submit = st.form_submit_button("Register", use_container_width=True)

                if reg_submit:
                    if user_exists(new_u):
                        st.error("User already exists")
                    elif new_u and new_p:
                        register_user(new_u, new_p)
                        st.success("Account Created! Please Login.")
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.warning("All fields are required")
            
            if st.button("Back to Login"):
                st.session_state.auth_page = "login"
                st.rerun()

    st.stop()
# ================= MAIN TABS =================
tab_home, tab_risk, tab_history, tab_profile = st.tabs(
    ["Home", "Risk Assessment", "History", "Profile"]
)

# ================= HOME =================
with tab_home:
    st.header("Heart Disease Risk Prediction System")
    st.write(
        "This application estimates the risk of heart disease using "
        "machine learning and basic clinical health indicators."
    )

    st.divider()

    if data is None:
        st.warning("Dataset not loaded. Home analytics unavailable.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Size", len(data))
    c2.metric("Heart Disease Cases", f"{data['HeartDisease'].mean()*100:.1f}%")
    c3.metric("Model Used", "Logistic Regresion")

    st.divider()

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Heart Disease Distribution**")
        st.bar_chart(data["HeartDisease"].value_counts())

    with col2:
        st.markdown("**Age Distribution**")
        bins = [20, 30, 40, 50, 60, 70, 80, 100]
        labels = ["20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80+"]
        age_groups = pd.cut(
            data["Age"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        st.bar_chart(age_groups.value_counts().sort_index())
        st.divider()

        # ================= SYSTEM OVERVIEW =================
    st.subheader("About This System")

    st.markdown(
        """
        This Heart Disease Risk Prediction System is a **clinical decision-support tool**
        designed to estimate the likelihood of heart disease based on commonly available
        health indicators.

        The system combines:
        - Medical knowledge (risk factors)
        - Historical patient data
        - A trained machine learning model

        to provide an **early risk screening**, not a diagnosis.
        """
    )

    st.divider()

    # ================= DATA USED =================
    st.subheader("What Data Is Used?")

    st.markdown(
        """
        The model is trained on a curated heart disease dataset containing
        real-world patient records.  
        The following types of information are used:
        """
    )

    st.markdown(
        """
        - **Demographics**: Age, Sex  
        - **Vital signs**: Blood pressure, heart rate  
        - **Blood test results**: Cholesterol, fasting blood sugar  
        - **Symptoms**: Chest pain, exercise-induced discomfort  
        - **Clinical indicators**: ECG results, stress test outcomes
        """
    )

    st.divider()

    # ================= FEATURE EXPLANATION =================
    st.subheader("Key Risk Factors Considered")

    st.markdown(
        """
        During prediction, the system automatically evaluates important
        cardiovascular risk factors, including:
        """
    )

    st.markdown(
        """
        - Age above 55 years  
        - High blood pressure (above 130 mmHg)  
        - High cholesterol levels (above 250 mg/dl)  
        - Elevated fasting blood sugar  

        These factors are combined into a **risk factor count** that supports
        the machine learning prediction.
        """
    )

    st.divider()

    # ================= MODEL EXPLANATION =================
    st.subheader("How Does the Prediction Work?")

    st.markdown(
        """
        1. You provide basic health information (or upload a report).  
        2. The system preprocesses the inputs and derives risk indicators.  
        3. A trained **Random Forest machine learning model** analyzes patterns
        learned from historical patient data.  
        4. The model outputs a **risk probability** between 0% and 100%.  
        5. The probability is categorized into **Low, Moderate, or High risk**.
        """
    )

    st.divider()







# ================= RISK =================
with tab_risk:
    st.header("Heart Risk Assessment")

    ocr = {}
    if OCR_AVAILABLE:
        with st.expander("Upload Medical Report (Optional)"):
            file = st.file_uploader("PDF / Image", type=["pdf", "png", "jpg"])
            if file:
                ocr = parse_report(extract_text(file))
                st.success("Report scanned. Please verify values.")

    age = st.number_input("Age", value=ocr.get("age") or 45)
    sex = 0 if st.radio("Gender", ["Female", "Male"]) == "Female" else 1
    bp = st.number_input("Systolic Blood Pressure", value=ocr.get("bp") or 120)
    chol = st.number_input("Total Cholesterol (mg/dl)", value=ocr.get("chol") or 200)
    fbs = 1 if st.radio("Fasting blood sugar above 120?", ["No", "Yes"]) == "Yes" else 0

    chest = {"No pain":0,"Mild":1,"Moderate":2,"Severe":3}[
        st.selectbox("Chest pain level", ["No pain", "Mild", "Moderate", "Severe"])
    ]
    angina = 1 if st.radio("Exercise-induced angina?", ["No", "Yes"]) == "Yes" else 0
    ecg = {"Normal":0,"Minor issue":1,"Major issue":2}[
        st.selectbox("ECG result", ["Normal", "Minor issue", "Major issue"])
    ]
    slope = {"Normal":2,"Flat":1,"Abnormal":0}[
        st.selectbox("Stress test result", ["Normal", "Flat", "Abnormal"])
    ]
    oldpeak = st.number_input("ST depression", value=0.0)
    maxhr = st.number_input("Max heart rate achieved", value=150)

    if st.button("Check My Risk", use_container_width=True):
        rf = (age > 55) + (bp > 130) + (chol > 250) + (fbs == 1)
        X = pd.DataFrame([{
            "Age": age, "Sex": sex, "ChestPainType": chest,
            "RestingBP": bp, "Cholesterol": chol, "FastingBS": fbs,
            "RestingECG": ecg, "MaxHR": maxhr,
            "ExerciseAngina": angina, "Oldpeak": oldpeak,
            "ST_Slope": slope, "risk_factor_count": rf
        }])

        prob = model.predict_proba(X)[0][1]
        risk = "High" if prob >= 0.7 or rf >= 3 else "Moderate" if prob >= 0.3 or rf == 2 else "Low"

        st.metric("Risk Probability", f"{prob*100:.2f}%")
        st.metric("Risk Category", risk)
        st.metric("Risk Factors", f"{rf}/4")
        st.write("DEBUG → Probability:", prob)
        st.write("DEBUG → Risk Factors:", rf)
        
        save_prediction(st.session_state.user, prob, risk, rf)

# HISTORY 
with tab_history:
    st.header("Prediction History")
    rows = get_user_predictions(st.session_state.user)
    if not rows:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(rows, columns=["probability","risk","risk_factors","date"])
        st.dataframe(df)

#  PROFILE
with tab_profile:
    st.header("User Profile")
    rows = get_user_predictions(st.session_state.user)

    if rows:
        df = pd.DataFrame(rows, columns=["probability","risk","risk_factors","date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        st.metric("Total Predictions", len(df))
        st.metric("Highest Risk", f"{df['probability'].max()*100:.1f}%")
        st.line_chart(df.set_index("date")["probability"] * 100)
    else:
        st.info("No activity yet.")

    if st.button("Logout"):
        st.session_state.logged = False
        st.session_state.auth_page = "login"
        st.rerun()
