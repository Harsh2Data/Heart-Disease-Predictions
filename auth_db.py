from database import get_connection
from datetime import datetime

def authenticate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cur.fetchone()
    conn.close()
    return user is not None

def user_exists(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def register_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
        (username, password, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
