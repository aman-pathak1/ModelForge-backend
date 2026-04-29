import sqlite3
import os

db_path = 'modelforge.db'
if not os.path.exists(db_path):
    print(f"Database file {db_path} not found.")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, email, verified FROM users")
        rows = cursor.fetchall()
        print("Registered Users:")
        print("-" * 50)
        print(f"{'ID':<40} | {'Email':<30} | {'Verified'}")
        print("-" * 50)
        for row in rows:
            print(f"{row[0]:<40} | {row[1]:<30} | {row[2]}")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
    finally:
        conn.close()
