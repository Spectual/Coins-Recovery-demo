import sqlite3
import os

def init_db(db_path):
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_image TEXT,
                missing_image TEXT,
                score INTEGER
            )
        ''')
        conn.commit()
        conn.close()

def save_match(db_path, api_image, missing_image, score):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO matches (api_image, missing_image, score) VALUES (?, ?, ?)',
              (api_image, missing_image, score))
    conn.commit()
    conn.close()