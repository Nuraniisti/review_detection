# save_to_db.py
import sqlite3
import os
from datetime import datetime

# Lokasi database
DB_PATH = os.path.join("db", "data_log.db")

# filepath: [save_to_db.py](http://_vscodecontentref_/5)

def lihat_riwayat(jumlah=20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM deteksi_ulasan ORDER BY id DESC LIMIT ?", (jumlah,))
    hasil = cursor.fetchall()
    conn.close()
    return hasil

def init_db():
    """
    Membuat folder db dan tabel deteksi_ulasan jika belum ada.
    """
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deteksi_ulasan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sumber TEXT,         -- manual / csv
            review TEXT,
            label TEXT,
            confidence REAL,
            prob_cg REAL,
            prob_or REAL,
            waktu TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_detection_record(review, label, confidence, prob_cg, prob_or, sumber="manual"):
    """
    Menyimpan satu hasil deteksi ke database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO deteksi_ulasan (sumber, review, label, confidence, prob_cg, prob_or, waktu)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sumber, review, label, float(confidence), float(prob_cg), float(prob_or), waktu))

    conn.commit()
    conn.close()


def get_all_records():
    """
    Mengambil semua data dari tabel deteksi_ulasan.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM deteksi_ulasan ORDER BY waktu DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "sumber": row[1],
            "review": row[2],
            "label": row[3],
            "confidence": row[4],
            "prob_cg": row[5],
            "prob_or": row[6],
            "waktu": row[7]
        }
        for row in rows
    ]
