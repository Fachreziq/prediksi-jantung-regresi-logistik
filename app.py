from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)

# Tiga berkas hasil notebook: model, scaler, dan urutan kolom saat pelatihan
model = joblib.load(BASE / "model_jantung.pkl")
scaler = joblib.load(BASE / "scaler_jantung.pkl")
KOLOM = joblib.load(BASE / "kolom_fitur.pkl")

# Kolom yang di-scaling di notebook (sama persis dengan sel 6)
FITUR_NUM = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
AMBANG = 0.5  # sama dengan ambang batas default saat evaluasi

# Batas keras input: (minimum, maksimum, nama untuk pesan error)
BATAS = {
    "Age": (1, 120, "Usia"),
    "RestingBP": (50, 260, "Tekanan darah istirahat"),
    "Cholesterol": (50, 700, "Kolesterol"),
    "MaxHR": (40, 250, "Detak jantung maksimum"),
    "Oldpeak": (-5, 8, "Oldpeak"),
}
# Rentang data latih (dari df.describe) untuk peringatan, bukan penolakan
RENTANG_DATA = {
    "Age": (28, 77, "Usia"),
    "RestingBP": (50, 200, "Tekanan darah istirahat"),
    "Cholesterol": (50, 603, "Kolesterol"),
    "MaxHR": (60, 202, "Detak jantung maksimum"),
    "Oldpeak": (-2.6, 6.2, "Oldpeak"),
}
PILIHAN = {
    "Sex": ({"M", "F"}, "Jenis kelamin"),
    "ChestPainType": ({"ATA", "NAP", "ASY", "TA"}, "Tipe nyeri dada"),
    "FastingBS": ({"0", "1"}, "Gula darah puasa"),
    "RestingECG": ({"Normal", "ST", "LVH"}, "Hasil EKG istirahat"),
    "ExerciseAngina": ({"Y", "N"}, "Nyeri dada saat olahraga"),
    "ST_Slope": ({"Up", "Flat", "Down"}, "Kemiringan segmen ST"),
}

# Hasil evaluasi dari notebook (data uji 184 pasien). Ubah bila model dilatih ulang.
JUMLAH_UJI = 184
METRIK = [
    ("Akurasi", "89%"),
    ("Recall", "91%"),
    ("Presisi", "89%"),
    ("AUC", "0,93"),
]

TEKS_PILIHAN = {
    "Sex": {"M": "Laki-laki", "F": "Perempuan"},
    "ChestPainType": {"TA": "TA, angina tipikal", "ATA": "ATA, angina atipikal",
                      "NAP": "NAP, nyeri non-angina", "ASY": "ASY, tanpa gejala"},
    "FastingBS": {"0": "Tidak (120 atau kurang)", "1": "Ya (di atas 120)"},
    "RestingECG": {"Normal": "Normal", "ST": "ST, kelainan gelombang ST-T",
                   "LVH": "LVH, pembesaran bilik kiri"},
    "ExerciseAngina": {"Y": "Ya", "N": "Tidak"},
    "ST_Slope": {"Up": "Naik", "Flat": "Datar", "Down": "Menurun"},
}


def ringkas_input(n):
    """Daftar (label, nilai) untuk ditampilkan kembali kepada pengguna."""
    return [
        ("Usia", f"{n['Age']:g} tahun"),
        ("Jenis kelamin", TEKS_PILIHAN["Sex"][n["Sex"]]),
        ("Tekanan darah istirahat", f"{n['RestingBP']:g} mm Hg"),
        ("Kolesterol serum", f"{n['Cholesterol']:g} mg/dl"),
        ("Detak jantung maksimum", f"{n['MaxHR']:g} per menit"),
        ("Gula darah puasa di atas 120", TEKS_PILIHAN["FastingBS"][n["FastingBS"]]),
        ("Tipe nyeri dada", TEKS_PILIHAN["ChestPainType"][n["ChestPainType"]]),
        ("EKG istirahat", TEKS_PILIHAN["RestingECG"][n["RestingECG"]]),
        ("Nyeri dada saat olahraga", TEKS_PILIHAN["ExerciseAngina"][n["ExerciseAngina"]]),
        ("Oldpeak", f"{n['Oldpeak']:g}"),
        ("Kemiringan segmen ST", TEKS_PILIHAN["ST_Slope"][n["ST_Slope"]]),
    ]


def baca_form(form):
    """Validasi input. Kembalikan (nilai_bersih, daftar_error)."""
    nilai, error = {}, []
    for kunci, (lo, hi, label) in BATAS.items():
        mentah = form.get(kunci, "").strip().replace(",", ".")
        try:
            angka = float(mentah)
        except ValueError:
            error.append(f"{label} harus berupa angka.")
            continue
        if not (lo <= angka <= hi):
            error.append(f"{label} harus antara {lo} dan {hi}.")
            continue
        nilai[kunci] = angka
    for kunci, (boleh, label) in PILIHAN.items():
        pilihan = form.get(kunci, "")
        if pilihan not in boleh:
            error.append(f"{label} belum dipilih.")
            continue
        nilai[kunci] = pilihan
    return nilai, error


def susun_fitur(n):
    """Ubah input pengguna menjadi baris fitur yang urutannya sama dengan saat pelatihan."""
    baris = {k: 0.0 for k in KOLOM}
    for k in FITUR_NUM:
        baris[k] = n[k]
    baris["FastingBS"] = float(n["FastingBS"])
    baris["Sex_M"] = 1.0 if n["Sex"] == "M" else 0.0
    baris["ExerciseAngina_Y"] = 1.0 if n["ExerciseAngina"] == "Y" else 0.0
    # Kategori pembanding (ASY, LVH, Down) memang tidak punya kolom: semua nol
    for kolom in (
        f"ChestPainType_{n['ChestPainType']}",
        f"RestingECG_{n['RestingECG']}",
        f"ST_Slope_{n['ST_Slope']}",
    ):
        if kolom in baris:
            baris[kolom] = 1.0
    X = pd.DataFrame([baris], columns=KOLOM)
    X[FITUR_NUM] = scaler.transform(X[FITUR_NUM])
    return X


@app.route("/", methods=["GET", "POST"])
def index():
    f = {}
    error = []
    peringatan = []
    hasil = None

    if request.method == "POST":
        f = request.form.to_dict()
        nilai, error = baca_form(request.form)
        if not error:
            proba = float(model.predict_proba(susun_fitur(nilai))[0][1])
            berisiko = proba >= AMBANG
            hasil = {
                "proba": proba,
                "persen": f"{proba * 100:.1f}".replace(".", ","),
                "lebar": round(proba * 100, 1),
                "berisiko": berisiko,
                "label": "Berisiko" if berisiko else "Tidak Berisiko",
                "persen_normal": f"{(1 - proba) * 100:.1f}".replace(".", ","),
                "ringkasan": ringkas_input(nilai),
            }
            for kunci, (lo, hi, label) in RENTANG_DATA.items():
                if not (lo <= nilai[kunci] <= hi):
                    peringatan.append(
                        f"{label} di luar rentang data latih ({lo:g} sampai {hi:g}), "
                        "sehingga prediksinya kurang dapat diandalkan."
                    )

    return render_template(
        "index.html", f=f, error=error, peringatan=peringatan, hasil=hasil,
        metrik=METRIK, jumlah_uji=JUMLAH_UJI,
    )


if __name__ == "__main__":
    app.run(debug=False)