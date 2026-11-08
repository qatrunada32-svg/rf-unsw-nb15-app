# Deteksi Anomali dan Serangan Jaringan dengan Random Forest (UNSW-NB15)

Ujian Tengah Semester, **Machine Learning for Network Data**, D4 Teknologi Rekayasa Komputer Jaringan, Universitas Bung Hatta.

| | |
|---|---|
| Nama | ISI NAMA LENGKAP |
| NPM | ISI NPM |
| Judul | Implementasi Random Forest untuk Klasifikasi Anomali dan Serangan pada Dataset UNSW-NB15 |
| Live app | ISI LINK https://....streamlit.app |
| Notebook Colab | ISI LINK COLAB (Anyone with the link can view) |

## Ringkasan
Dua model Random Forest dilatih di Google Colab pada UNSW-NB15 (175.341 flow latih, 82.332 flow test, 42 fitur):
1. **Model biner**: normal vs serangan.
2. **Model multikelas**: 10 kelas `attack_cat` (Normal + 9 kategori serangan).

Preprocessing (OneHotEncoder untuk `proto`, `service`, `state`) berada di dalam pipeline, sehingga aplikasi cukup memberi data mentah. Dataset dibagi memakai file train dan test bawaan UNSW-NB15, sehingga tidak ada kebocoran data.

## Hasil model (data test, dari notebook saya)
> Isi dengan angka dari run Colab Anda sendiri (output sel bagian 5, 6, dan 8).

| Model | Akurasi | Recall serangan | Precision | F1 | ROC-AUC | Waktu latih |
|---|---|---|---|---|---|---|
| Biner | ... | ... | ... | ... | ... | ... |
| Multikelas (10 kelas) | ... | macro F1 ... | | weighted F1 ... | | ... |

### Eksperimen hyperparameter (model biner)
> Salin tabel dari sel "Eksperimen hyperparameter" di notebook.

| Variasi | Accuracy | Recall | F1 | ROC-AUC | Waktu latih (s) | Ukuran model (MB) |
|---|---|---|---|---|---|---|
| A. Baseline | ... | ... | ... | ... | ... | ... |
| B. Dangkal (depth=10) | ... | ... | ... | ... | ... | ... |
| C. 200 pohon | ... | ... | ... | ... | ... | ... |
| D. class_weight=balanced | ... | ... | ... | ... | ... | ... |
| E. depth=30, leaf=1 | ... | ... | ... | ... | ... | ... |

## Modifikasi dari template aplikasi
1. **Evaluasi lengkap untuk CSV berlabel**: bila CSV punya kolom `label`, aplikasi menampilkan precision, recall, jumlah alarm palsu (FP), serangan lolos (FN), dan heatmap confusion matrix. Template hanya menampilkan akurasi.
2. **Confusion matrix kategori**: bila CSV juga punya `attack_cat`, kategori prediksi dibandingkan dengan kategori asli.
3. **Filter hasil per flow**: tampilkan hanya serangan atau pilih kategori tertentu; unduhan CSV mengikuti filter.
4. **Penjelasan kategori serangan dan saran tindak lanjut**: muncul setelah prediksi manual dan di tab baru **Kamus serangan**.
5. **Riwayat prediksi manual** (`st.session_state`) yang bisa diunduh atau dihapus.
6. **Tema warna sendiri** lewat `.streamlit/config.toml`.

## Struktur repo
```
app.py                 aplikasi Streamlit
requirements.txt       versi library sama dengan Colab (dari artifacts.zip)
model/                 rf_binary.joblib, rf_multiclass.joblib, metadata.json
data/sample_flows.csv  300 flow contoh dari data test
notebooks/             notebook training (Colab)
.streamlit/config.toml tema warna
```

## Menjalankan lokal
```bash
python -m venv .venv          # pakai Python 3.12
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Penggunaan AI
Sesuai aturan proyek, penggunaan AI dicatat di sini (sesuaikan dengan yang benar-benar Anda lakukan).

| Tool | Dipakai untuk |
|---|---|
| Claude (versi gratis) | Membantu menulis kode modifikasi `app.py`, sel eksperimen hyperparameter dan sel analisis di notebook |

Contoh prompt penting:
1. *ISI prompt Anda*
2. *ISI prompt Anda*

Seluruh angka hasil model pada README dan laporan berasal dari run notebook saya sendiri.

## Sitasi dataset
Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set).* Military Communications and Information Systems Conference (MilCIS).
