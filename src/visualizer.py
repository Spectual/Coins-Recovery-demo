import sqlite3
import matplotlib.pyplot as plt

def visualize_matches(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT api_image, missing_image, score FROM matches')
    results = c.fetchall()
    conn.close()

    scores = [r[2] for r in results]
    labels = [f"{r[0]} vs {r[1]}" for r in results]

    plt.figure(figsize=(12, 6))
    plt.barh(labels, scores)
    plt.xlabel('Matching Score')
    plt.title('Matching Results between API Images and Missing Images')
    plt.tight_layout()
    plt.show()