# Prediksi Risiko Penyakit Jantung (Regresi Logistik)

Aplikasi web Flask yang memprediksi risiko penyakit jantung dari 11 data klinis pasien
menggunakan Regresi Logistik (scikit-learn). Hasil berupa kategori
"Berisiko" / "Tidak Berisiko" beserta probabilitasnya.

- Dataset: Heart Failure Prediction (Kaggle, fedesoriano), 918 data
- Target: HeartDisease (1 = sakit jantung, 0 = normal)
- Model, scaler, dan daftar kolom disimpan dengan Joblib

Catatan: prototipe untuk tugas kuliah, bukan alat diagnosis medis.

## Menjalankan secara lokal
pip install -r requirements.txt
python app.py
Buka http://127.0.0.1:5000

## Deploy (Railway/Render)
Start command: gunicorn app:app --bind 0.0.0.0:$PORT