import sqlite3

def get_user_by_id(user_id):
    """Safe: Using parameterized queries."""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
