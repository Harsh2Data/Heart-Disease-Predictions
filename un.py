import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(
    page_title="Heart Disease Risk Prediction System",
    layout="wide"
)

USER_FILE = "users.csv"
PRED_FILE = "predictions.csv"

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)

if not os.path.exists(PRED_FILE):
    pd.DataFrame(
        columns=["username", "date", "probability", "risk", "risk_factors"]
    ).to_csv(PRED_FILE, index=False)

# ================= LOAD MODEL & DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv") if os.path.exists("heart.csv") else None

data = load_data()
model = joblib.load("model25.pkl")

# ================= HELPERS =================
def authenticate(u, p):
    df = pd.read_csv(USER_FILE)
    return not df[(df.username == u) & (df.password == p)].empty

def user_exists(u):
    df = pd.read_csv(USER_FILE)
    return not df[df.username == u].empty

def register_user(u, p):
    df = pd.read_csv(USER_FILE)
    df.loc[len(df)] = [u, p]
    df.to_csv(USER_FILE, index=False)

def save_prediction(u, prob, risk, rf):
    df = pd.read_csv(PRED_FILE)
    df.loc[len(df)] = [u, datetime.now(), prob, risk, rf]
    df.to_csv(PRED_FILE, index=False)

# ================= SESSION STATE =================
if "logged" not in st.session_state:
    st.session_state.logged = False

if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"   # login | register

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ================= AUTH =================
if not st.session_state.logged:

    st.title("Heart Disease Risk Prediction System")
    st.caption("Login to continue")

    if st.session_state.auth_view == "login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if authenticate(username, password):
                st.session_state.logged = True
                st.session_state.user = username
                st.session_state.page = "Home"   # FORCE HOME
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("----")
        if st.button("New user? Register here"):
            st.session_state.auth_view = "register"
            st.experimental_rerun()

    else:  # REGISTER VIEW
        st.subheader("Create New Account")

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register", use_container_width=True):
            if user_exists(new_user):
                st.error("Username already exists")
            elif new_user and new_pass:
                register_user(new_user, new_pass)
                st.success("Registration successful. Please login.")
                st.session_state.auth_view = "login"
                st.experimental_rerun()
            else:
                st.warning("Enter username and password")

        if st.button("Back to Login"):
            st.session_state.auth_view = "login"
            st.experimental_rerun()

    st.stop()

# ================= SIDEBAR =================
st.session_state.page = st.sidebar.radio(
    "Navigation",
    ["Home", "Risk Assessment", "History", "Profile"],
    index=["Home", "Risk Assessment", "History", "Profile"].index(st.session_state.page)
)

# ================= HOME (FIXED DASHBOARD) =================
if st.session_state.page == "Home":
    st.header("Dashboard Overview")
    st.write(
        "This system estimates heart disease risk using clinical indicators and symptoms."
    )

    if data is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records Used", len(data))
        c2.metric("Heart Disease Rate", f"{data.HeartDisease.mean()*100:.1f}%")
        c3.metric("Model Used", "Random Forest")

# ================= RISK ASSESSMENT =================
if st.session_state.page == "Risk Assessment":
    st.header("Heart Risk Assessment")

    age = st.number_input("Age", value=45)
    sex = st.radio("Gender", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    bp = st.number_input("Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)

    fbs = st.radio("Fasting blood sugar above 120?", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

    chest = st.selectbox("Chest pain level", ["No pain", "Mild", "Moderate", "Severe"])
    chest = {"No pain": 0, "Mild": 1, "Moderate": 2, "Severe": 3}[chest]

    angina = st.radio("Pain during exercise?", ["No", "Yes"])
    angina = 1 if angina == "Yes" else 0

    oldpeak = st.number_input("ST depression", value=0.0)
    maxhr = st.number_input("Max heart rate", value=150)

    if st.button("Predict Risk", use_container_width=True):

        rf = (age > 55) + (bp > 130) + (chol > 250) + (fbs == 1)

        X = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest,
            "RestingBP": bp,
            "Cholesterol": chol,
            "FastingBS": fbs,
            "RestingECG": 0,
            "MaxHR": maxhr,
            "ExerciseAngina": angina,
            "Oldpeak": oldpeak,
            "ST_Slope": 2,
            "risk_factor_count": rf
        }])

        prob = model.predict_proba(X)[0][1]

        if prob >= 0.7 or rf >= 3:
            risk = "High"
        elif prob >= 0.3 or rf == 2:
            risk = "Moderate"
        else:
            risk = "Low"

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Probability", f"{prob*100:.2f}%")
        c2.metric("Risk Category", risk)
        c3.metric("Risk Factors", f"{rf}/4")

        save_prediction(st.session_state.user, prob, risk, rf)

# ================= HISTORY =================
if st.session_state.page == "History":
    st.header("Prediction History")
    df = pd.read_csv(PRED_FILE)
    st.dataframe(df[df.username == st.session_state.user])

# ================= PROFILE =================
if st.session_state.page == "Profile":
    st.header("Profile")
    st.write("Logged in as:", st.session_state.user)
    if st.button("Logout"):
        st.session_state.logged = False
        st.session_state.auth_view = "login"
        st.experimental_rerun()
