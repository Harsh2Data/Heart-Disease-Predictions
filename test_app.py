import streamlit as st
import pandas as pd
import joblib
import os
import re
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime
from PIL import Image
from io import BytesIO

# ================= 1. CONFIGURATION =================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="➕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. PROFESSIONAL DARK THEME CSS =================
st.markdown("""
<style>
    /* 1. Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* 2. Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }

    /* 3. Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1F2937;
        border-radius: 5px 5px 0 0;
        color: #FFFFFF;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid #374151;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border-color: #FF4B4B;
    }

    /* 4. Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }
    p, label {
        color: #BFC5C9 !important;
    }

    /* 5. Inputs (Fixing white background issues) */
    div[data-testid="stTextInput"] input, 
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stNumberInput"] div {
        color: #FFFFFF !important;
        background-color: #262730 !important;
    }
    
    /* 6. Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] { color: #9CA3AF !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }

    /* 7. Buttons */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    /* 8. Tables */
    div[data-testid="stDataFrame"] {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. DATABASE FUNCTIONS =================
DB_NAME = "heart_app.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, 
        password TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, 
        probability REAL, 
        risk TEXT, 
        risk_factors INTEGER, 
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# --- Auth Functions ---
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
    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username=? AND password=?", (username, password))
    ok = cur.fetchone() is not None
    conn.close()
    return ok

def save_prediction(username, prob, risk, rf):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (username, probability, risk, risk_factors, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (username, prob, risk, rf, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_user_predictions(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT probability, risk, risk_factors, created_at
        FROM predictions WHERE username=? ORDER BY created_at DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ================= 4. UTILITIES (PDF, OCR, ADVICE) =================

# --- A. PDF GENERATOR ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

def create_pdf_report(user, date, risk, prob, advice_list, vitals):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Medical Risk Assessment Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Patient Name: {user}")
    c.drawString(50, height - 100, f"Date: {date}")
    c.line(50, height - 110, width - 50, height - 110)
    
    # Result Box
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 150, f"Risk Level: {risk}")
    c.drawString(50, height - 175, f"Probability Score: {prob*100:.1f}%")
    
    # Vitals
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 220, "Recorded Vitals:")
    c.setFont("Helvetica", 12)
    y = height - 245
    for k, v in vitals.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 20
        
    # Recommendations
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Clinical Recommendations:")
    c.setFont("Helvetica", 12)
    y -= 25
    for item in advice_list:
        c.drawString(70, y, f"- {item}")
        y -= 20
        
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Disclaimer: This report is generated by AI and does not replace professional medical diagnosis.")
    c.save()
    buffer.seek(0)
    return buffer

# --- B. OCR ---
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def extract_text(file):
    if not OCR_AVAILABLE: return ""
    try:
        if file.type == "application/pdf":
            images = convert_from_bytes(file.read())
            return " ".join(pytesseract.image_to_string(img) for img in images)
        return pytesseract.image_to_string(Image.open(file))
    except: return ""

def parse_report(text):
    def find(p, t): 
        m = re.search(p, t, re.IGNORECASE)
        return int(m.group(1)) if m else None
    return {
        "age": find(r"Age[:\s]+(\d+)", text),
        "bp": find(r"Blood\s*Pressure[:\s]+(\d+)", text),
        "chol": find(r"Cholesterol[:\s]+(\d+)", text),
        "fbs": find(r"Fasting\s*Blood\s*Sugar[:\s]+(\d+)", text),
    }

# --- C. MEDICAL LOGIC ---
def generate_advice(bp, chol, fbs, bmi, age):
    advice = []
    if bp > 130: advice.append("Elevated Blood Pressure. Reduce sodium, manage stress.")
    if chol > 200: advice.append("High Cholesterol. Adopt a heart-healthy diet low in saturated fats.")
    if fbs == 1: advice.append("High Fasting Blood Sugar. Consult endocrinologist regarding pre-diabetes.")
    if bmi > 25: advice.append("BMI indicates overweight. Aim for 150 mins of moderate activity per week.")
    if age > 50: advice.append("Age-related risk factor. Regular cardiac screenings advised.")
    if not advice: advice.append("Vitals are within healthy range. Maintain current lifestyle.")
    return advice

# ================= 5. LOAD DATA & MODEL =================
@st.cache_data
def load_data():
    if os.path.exists("heart.csv"):
        return pd.read_csv("heart.csv")
    return None

data = load_data()

try:
    if os.path.exists("model25.pkl"):
        model = joblib.load("model25.pkl")
    else:
        model = None
except:
    model = None

# ================= 6. AUTHENTICATION FLOW =================
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

# ================= 7. MAIN APPLICATION =================

# --- SIDEBAR ---
with st.sidebar:
    st.title("Heart Disease System")
    st.markdown(f"User: **{st.session_state.user}**")
    st.markdown("---")
    if st.button("Log Out"):
        st.session_state.logged = False
        st.session_state.auth_page = "login"
        st.rerun()

# --- TABS ---
tab_dash, tab_assess, tab_hist, tab_prof = st.tabs([
    "Dashboard", 
    "Prediction", 
    "History", 
    "Profile"
])

# --- TAB 1: DASHBOARD ---
with tab_dash:
    st.title("Clinical Dashboard")
    st.markdown("Overview of heart disease metrics and system status.")
    
    if data is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Patients", len(data))
        m2.metric("Avg Age", f"{int(data['Age'].mean())}")
        m3.metric("High Risk %", f"{data['HeartDisease'].mean()*100:.1f}%")
        m4.metric("System Status", "Online")
        
        st.markdown("### Analytics")
        c1, c2 = st.columns(2)
        
        # CHART 1: DONUT PIE
        with c1:
            st.markdown("**Disease Distribution**")
            count_1 = data[data['HeartDisease'] == 1].shape[0]
            count_0 = data[data['HeartDisease'] == 0].shape[0]
            
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0E1117') 
            ax.set_facecolor('#0E1117')
            
            ax.pie([count_1, count_0], labels=['Heart Disease', 'Normal'], 
                   autopct='%1.1f%%', startangle=90, 
                   colors=['#FF4B4B', '#4B90FF'], 
                   textprops={'color':"white", 'weight':'bold'})
            
            # Donut Hole
            fig.gca().add_artist(plt.Circle((0,0),0.70,fc='#0E1117'))
            st.pyplot(fig, use_container_width=True)

        # CHART 2: DENSITY CURVE
        with c2:
            st.markdown("**Age Distribution (Density)**")
            ages = data['Age'].dropna()
            
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            # Histogram
            ax.hist(ages, bins=15, density=True, alpha=0.5, color='#4B90FF', edgecolor='none')
            
            # Curve
            try:
                density = gaussian_kde(ages)
                xs = np.linspace(min(ages), max(ages), 200)
                ax.plot(xs, density(xs), color='#00d4ff', linewidth=3)
            except: pass
            
            ax.set_xlabel("Age", color="white")
            ax.tick_params(colors="white")
            ax.grid(color='#444444', linestyle='--', linewidth=0.5, alpha=0.3)
            for spine in ax.spines.values(): spine.set_edgecolor('#444444')
                
            st.pyplot(fig, use_container_width=True)

    else:
        st.warning("Dataset (heart.csv) not found. Metrics unavailable.")

# --- TAB 2: PREDICTION ---
with tab_assess:
    st.title("Heart Disease Prediction")
    
    if model is None:
        st.error("⚠️ Model file (model25.pkl) not found. Cannot run predictions.")
    
    ocr = {}
    if OCR_AVAILABLE:
        with st.expander("Upload Medical Report (PDF/Image)"):
            file = st.file_uploader("File", type=["pdf", "png", "jpg"])
            if file:
                ocr = parse_report(extract_text(file))
                st.success("Values extracted.")

    with st.form("risk_form"):
        st.subheader("1. Vitals & BMI")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=1, value=ocr.get("age") or 45)
            sex = 1 if st.selectbox("Gender", ["Male", "Female"]) == "Male" else 0
            height = st.number_input("Height (cm)", 100, 250, 170)
        with c2:
            bp = st.number_input("Resting BP", value=ocr.get("bp") or 120)
            chol = st.number_input("Cholesterol", value=ocr.get("chol") or 200)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
        with c3:
            maxhr = st.number_input("Max Heart Rate", value=150)
            fbs = 1 if st.selectbox("Fasting Sugar > 120?", ["No", "Yes"]) == "Yes" else 0
            # BMI Calc
            bmi = weight / ((height/100)**2)
            st.metric("Calculated BMI", f"{bmi:.1f}")

        st.subheader("2. Clinical")
        c4, c5 = st.columns(2)
        with c4:
            chest = {"No pain":0,"Mild":1,"Moderate":2,"Severe":3}[
                st.selectbox("Chest Pain", ["No pain", "Mild", "Moderate", "Severe"])
            ]
            ecg = {"Normal":0,"Minor issue":1,"Major issue":2}[
                st.selectbox("ECG", ["Normal", "Minor issue", "Major issue"])
            ]
        with c5:
            angina = 1 if st.radio("Exercise Angina?", ["No", "Yes"], horizontal=True) == "Yes" else 0
            slope = {"Normal":2,"Flat":1,"Abnormal":0}[
                st.selectbox("ST Slope", ["Normal", "Flat", "Abnormal"])
            ]
            oldpeak = st.number_input("ST Depression", value=0.0)

        run_pred = st.form_submit_button("Analyze Risk", use_container_width=True)

    if run_pred:
        if model:
            # Prepare Input
            rf = (age > 55) + (bp > 130) + (chol > 200) + (fbs == 1) + (bmi > 25)
            X = pd.DataFrame([{
                "Age": age, "Sex": sex, "ChestPainType": chest,
                "RestingBP": bp, "Cholesterol": chol, "FastingBS": fbs,
                "RestingECG": ecg, "MaxHR": maxhr,
                "ExerciseAngina": angina, "Oldpeak": oldpeak,
                "ST_Slope": slope, "risk_factor_count": rf
            }])
            
            # Predict
            prob = model.predict_proba(X)[0][1]
            
            # Categories
            if prob >= 0.7: 
                risk="High"; color="red"
            elif prob >= 0.3: 
                risk="Moderate"; color="orange"
            else: 
                risk="Low"; color="green"

            save_prediction(st.session_state.user, prob, risk, rf)

            # Display Result
            st.markdown(f"""
            <div style="background-color: #1F2937; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-top: 20px;">
                <h2 style="color: {color}; margin:0;">{risk} Risk Detected</h2>
                <p style="color: white; font-size: 18px;">Probability: {prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 AI Recommendations")
            advice = generate_advice(bp, chol, fbs, bmi, age)
            for item in advice:
                st.info(item)

            # Download PDF
            if PDF_AVAILABLE:
                vitals_dict = {"Age": age, "BP": bp, "Cholesterol": chol, "BMI": f"{bmi:.1f}"}
                pdf_data = create_pdf_report(st.session_state.user, datetime.now().strftime("%Y-%m-%d"), risk, prob, advice, vitals_dict)
                st.download_button(label="📄 Download Medical Report", data=pdf_data, file_name="Heart_Report.pdf", mime="application/pdf")
        else:
            st.error("Model not loaded. Please ensure 'model25.pkl' is in the folder.")

# --- TAB 3: HISTORY ---
with tab_hist:
    st.title("Patient History")
    rows = get_user_predictions(st.session_state.user)
    if rows:
        df = pd.DataFrame(rows, columns=["Probability", "Risk", "Factors", "Date"])
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records found.")

# --- TAB 4: PROFILE ---
with tab_prof:
    st.title("User Profile")
    rows = get_user_predictions(st.session_state.user)
    
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f"""
        <div style="background-color: #262730; padding: 20px; border-radius: 10px;">
            <h3>{st.session_state.user}</h3>
            <p>Assessments: {len(rows)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        if rows:
            df = pd.DataFrame(rows, columns=["Probability", "Risk", "Factors", "Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            st.subheader("Risk Trend")
            st.line_chart(df.set_index("Date")["Probability"])
        else:
            st.info("No data available for charts.")