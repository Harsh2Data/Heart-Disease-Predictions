# database.py
import sqlite3
from datetime import datetime

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
