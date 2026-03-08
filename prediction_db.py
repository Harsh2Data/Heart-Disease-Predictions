from database import get_connection
from datetime import datetime

def save_prediction(username, probability, risk, risk_factors):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions
        (username, probability, risk, risk_factors, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, probability, risk, risk_factors, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_user_predictions(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT probability, risk, risk_factors, created_at
        FROM predictions
        WHERE username=?
        ORDER BY created_at
        """,
        (username,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows
