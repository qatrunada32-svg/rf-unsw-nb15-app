"""
Aplikasi Streamlit: Klasifikasi Anomali dan Serangan Jaringan (UNSW-NB15) dengan Random Forest
Ujian Tengah Semester, Machine Learning for Network Data, D4 TRKJ Universitas Bung Hatta

Dasar: template Proyek 01 dari dosen. Modifikasi yang saya tambahkan (lihat juga README.md):
  1. Confusion matrix + precision/recall/alarm palsu/serangan lolos bila CSV punya kolom `label`.
  2. Perbandingan kategori prediksi vs `attack_cat` asli bila CSV punya kolom `attack_cat`.
  3. Filter hasil per flow (hanya serangan, per kategori).
  4. Penjelasan tiap kategori serangan + saran tindak lanjut (hasil prediksi manual dan tab Kamus).
  5. Riwayat prediksi manual (st.session_state) yang bisa diunduh.
  6. Tema warna sendiri (.streamlit/config.toml).

Jalankan lokal:  streamlit run app.py
"""
import json
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# ===== ISI IDENTITAS ANDA =====
NAMA = "CUT QATRUNADA NASYWA MAHADEWI"
NPM = "2410017514008"

BASE = Path(__file__).parent
MODEL_DIR = BASE
SAMPLE_CSV = BASE / "sample_flows.csv"

st.set_page_config(page_title="Deteksi Serangan UNSW-NB15", page_icon="🛡️", layout="wide")

# Penjelasan kategori serangan UNSW-NB15 (Moustafa & Slay, 2015) dan saran tindak lanjut singkat
KAMUS_SERANGAN = {
    "Normal": ("Trafik wajar tanpa indikasi serangan.",
               "Tidak perlu tindakan. Bila ini alarm palsu, naikkan threshold di sidebar."),
    "Generic": ("Teknik yang bekerja melawan semua block cipher tanpa memandang struktur cipher-nya.",
                "Gunakan algoritma enkripsi dan panjang kunci modern, dan pantau pola query yang tidak wajar."),
    "Exploits": ("Memanfaatkan bug atau kerentanan pada sistem, layanan, atau perangkat lunak yang sudah dikenal.",
                 "Pasang patch keamanan, perbarui perangkat lunak, dan batasi layanan yang terbuka ke jaringan."),
    "Fuzzers": ("Mengirim data acak atau cacat ke layanan untuk membuatnya crash dan menemukan celah.",
                "Validasi input di sisi server, batasi laju koneksi, dan aktifkan IDS/IPS."),
    "DoS": ("Membanjiri layanan dengan permintaan sehingga tidak bisa melayani pengguna sah.",
            "Terapkan rate limiting, firewall, dan load balancing, serta siapkan mitigasi dari ISP."),
    "Reconnaissance": ("Pengintaian: memindai jaringan atau port untuk mengumpulkan informasi sebelum menyerang.",
                       "Tutup port yang tidak dipakai, blokir sumber pemindai, dan catat alamat sumbernya."),
    "Analysis": ("Gabungan serangan seperti port scan, spam, dan penetrasi lewat file HTML.",
                 "Filter spam dan konten web, serta periksa log layanan web."),
    "Backdoor": ("Membuat jalur akses tersembunyi yang melewati autentikasi normal.",
                 "Audit akun dan layanan yang berjalan, periksa koneksi keluar yang janggal, ganti kredensial."),
    "Shellcode": ("Payload kecil yang dikirim lewat eksploitasi untuk mendapatkan akses shell pada target.",
                  "Isolasi host yang dicurigai, perbarui sistem, dan aktifkan proteksi memori (DEP/ASLR)."),
    "Worms": ("Malware yang menggandakan diri dan menyebar otomatis ke host lain di jaringan.",
              "Segmentasi jaringan, isolasi host terinfeksi, dan perbarui antivirus serta patch."),
}


# ---------- Muat model (sekali saja, lalu di-cache) ----------
@st.cache_resource
def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    rf_bin = joblib.load(MODEL_DIR / "rf_binary.joblib")
    rf_cat = joblib.load(MODEL_DIR / "rf_multiclass.joblib")
    return rf_bin, rf_cat, meta


rf_bin, rf_cat, meta = load_models()
FEATURES = meta["features"]


def predict(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Lengkapi kolom yang hilang dengan nilai default, lalu prediksi dengan kedua model."""
    X = df.copy()
    for col in FEATURES:
        if col not in X.columns:
            X[col] = meta["defaults"][col]
    X = X[FEATURES]
    for col in meta["cat_cols"]:
        X[col] = X[col].astype(str)
    prob = rf_bin.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "prob_serangan": prob.round(3),
        "prediksi": ["SERANGAN" if p >= threshold else "normal" for p in prob],
        "kategori_prediksi": rf_cat.predict(X),
    })
    return out


def heatmap_confusion(cm: pd.DataFrame, judul: str, tinggi: int = 280) -> alt.Chart:
    """Heatmap confusion matrix (Altair, sudah tersedia bersama Streamlit)."""
    long = cm.reset_index().melt(id_vars=cm.index.name, var_name="Prediksi", value_name="Jumlah")
    long = long.rename(columns={cm.index.name: "Aktual"})
    urutan = list(cm.columns)
    base = alt.Chart(long, title=judul).encode(
        x=alt.X("Prediksi:N", sort=urutan, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Aktual:N", sort=list(cm.index)))
    rect = base.mark_rect().encode(color=alt.Color("Jumlah:Q", scale=alt.Scale(scheme="blues"), legend=None))
    teks = base.mark_text(fontSize=13).encode(
        text="Jumlah:Q",
        color=alt.condition(alt.datum.Jumlah > long["Jumlah"].max() / 2, alt.value("white"), alt.value("black")))
    return (rect + teks).properties(height=tinggi)


def kartu_kategori(kat: str):
    """Tampilkan penjelasan dan saran untuk satu kategori serangan."""
    if kat in KAMUS_SERANGAN:
        arti, saran = KAMUS_SERANGAN[kat]
        st.markdown(f"**Apa itu {kat}?** {arti}")
        st.markdown(f"**Saran tindak lanjut:** {saran}")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold = st.slider("Threshold probabilitas serangan", 0.05, 0.95, 0.50, 0.05,
                          help="Flow dianggap serangan bila probabilitasnya ≥ threshold. "
                               "Turunkan untuk menangkap lebih banyak serangan (recall naik), "
                               "dengan risiko alarm palsu bertambah.")
    st.divider()
    st.subheader("Performa model (data test)")
    mb, mc = meta["metrics_binary"], meta["metrics_multiclass"]
    c1, c2 = st.columns(2)
    c1.metric("Akurasi biner", f"{mb['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{mb['roc_auc']:.3f}")
    c1.metric("Recall serangan", f"{mb['recall']:.1%}")
    c2.metric("Akurasi kategori", f"{mc['accuracy']:.1%}")
    st.caption(f"Dataset: {meta['dataset']}")
    st.divider()
    st.caption(f"UTS Machine Learning for Network Data\n\n**{NAMA}** ({NPM})\n\nD4 TRKJ, Universitas Bung Hatta")

# ---------- Halaman utama ----------
st.title("🛡️ Deteksi Anomali dan Serangan Jaringan")
st.write("Model **Random Forest** yang dilatih pada dataset **UNSW-NB15** memprediksi apakah sebuah "
         "flow jaringan normal atau serangan, sekaligus menebak kategori serangannya.")

tab_csv, tab_manual, tab_kamus, tab_info = st.tabs(
    ["📄 Prediksi dari CSV", "✍️ Input manual", "📚 Kamus serangan", "ℹ️ Tentang model"])

# --- Tab 1: CSV ---
with tab_csv:
    st.write("Unggah file CSV berisi flow dengan kolom seperti dataset UNSW-NB15, "
             "atau pakai 300 flow contoh dari data test.")
    uploaded = st.file_uploader("File CSV", type="csv")
    use_sample = st.toggle("Gunakan data contoh", value=uploaded is None)

    df = None
    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv(SAMPLE_CSV)

    if df is not None:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} kolom tidak ada dan diisi nilai default (median data latih): "
                       f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        res = predict(df, threshold)
        shown = pd.concat([res, df.drop(columns=[c for c in res.columns if c in df.columns])], axis=1)
        is_attack = res["prediksi"] == "SERANGAN"

        n_att = int(is_attack.sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Jumlah flow", len(res))
        k2.metric("Diprediksi serangan", n_att)
        k3.metric("Persentase serangan", f"{n_att / len(res):.1%}")

        # [Modifikasi 1] Evaluasi lengkap bila ada label asli
        if "label" in df.columns:
            y_true = df["label"].astype(int)
            y_pred = is_attack.astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            acc = (tp + tn) / len(res)
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")

            st.info(f"File memuat kolom `label`, jadi prediksi bisa dicek: akurasi biner = **{acc:.1%}** "
                    f"(threshold {threshold:.2f})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", "-" if pd.isna(prec) else f"{prec:.1%}")
            m2.metric("Recall serangan", "-" if pd.isna(rec) else f"{rec:.1%}")
            m3.metric("Alarm palsu (FP)", fp, help="Flow normal yang dikira serangan")
            m4.metric("Serangan lolos (FN)", fn, help="Serangan yang dikira normal")

            cm_bin = pd.DataFrame([[tn, fp], [fn, tp]],
                                  index=pd.Index(["normal", "serangan"], name="Aktual"),
                                  columns=["normal", "serangan"])
            cm_col, cat_col = st.columns(2)
            with cm_col:
                st.altair_chart(heatmap_confusion(cm_bin, "Confusion matrix biner"), width="stretch")

            # [Modifikasi 2] Bandingkan kategori prediksi dengan attack_cat asli
            if "attack_cat" in df.columns:
                acc_cat = (res["kategori_prediksi"] == df["attack_cat"]).mean()
                with cat_col:
                    cm_cat = pd.crosstab(df["attack_cat"].rename("Aktual"),
                                         res["kategori_prediksi"].rename("Prediksi"))
                    st.altair_chart(heatmap_confusion(cm_cat, f"Kategori (akurasi {acc_cat:.1%})", 280),
                                    width="stretch")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Kategori yang diprediksi")
            kat = res["kategori_prediksi"].value_counts().rename_axis("kategori").reset_index(name="jumlah")
            st.bar_chart(kat, x="kategori", y="jumlah", horizontal=True, sort="-jumlah")
        with right:
            st.subheader("Sebaran probabilitas serangan")
            st.bar_chart(pd.cut(res["prob_serangan"], bins=[0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
                         .astype(str).value_counts().sort_index())

        # [Modifikasi 3] Filter tabel hasil
        st.subheader("Hasil per flow")
        opsi_kat = sorted(res["kategori_prediksi"].unique())
        f1, f2 = st.columns([1, 3])
        hanya_serangan = f1.checkbox("Hanya serangan", value=False)
        pilih_kat = f2.multiselect("Filter kategori prediksi", opsi_kat, default=opsi_kat)
        mask = res["kategori_prediksi"].isin(pilih_kat)
        if hanya_serangan:
            mask &= is_attack
        view = shown[mask]
        st.caption(f"Menampilkan {len(view)} dari {len(shown)} flow")
        st.dataframe(view, width="stretch", height=360)
        st.download_button("⬇️ Unduh hasil yang ditampilkan (CSV)", view.to_csv(index=False).encode(),
                           "hasil_prediksi.csv", "text/csv")

# --- Tab 2: Input manual ---
with tab_manual:
    st.write("Pilih flow contoh sebagai titik awal, lalu ubah fitur-fitur terpenting dan tekan **Prediksi**. "
             "Fitur yang tidak ditampilkan memakai nilai dari flow contoh tersebut.")
    samples = pd.read_csv(SAMPLE_CSV)
    pilihan = st.selectbox("Mulai dari contoh flow kategori", sorted(samples["attack_cat"].unique()),
                           index=sorted(samples["attack_cat"].unique()).index("Normal"))
    base = samples[samples["attack_cat"] == pilihan].iloc[0]
    with st.form("manual"):
        cols = st.columns(3)
        row = {c: base[c] for c in FEATURES}
        for i, c in enumerate(meta["cat_cols"]):
            opts = sorted(set(meta["cat_options"][c]) | {str(base[c])})
            row[c] = cols[i].selectbox(c, opts, index=opts.index(str(base[c])), key=f"{pilihan}-{c}")
        cols = st.columns(4)
        for i, c in enumerate(meta["top_numeric"]):
            lo, hi = meta["num_ranges"][c]
            row[c] = cols[i % 4].number_input(c, min_value=float(lo), value=float(base[c]), key=f"{pilihan}-{c}",
                                              help=f"Rentang umum data latih: {lo:g} sampai {hi:g}")
        submitted = st.form_submit_button("🔍 Prediksi", type="primary")

    if "riwayat" not in st.session_state:
        st.session_state["riwayat"] = []

    if submitted:
        r = predict(pd.DataFrame([row]), threshold).iloc[0]
        if r["prediksi"] == "SERANGAN":
            st.error(f"**SERANGAN** terdeteksi (probabilitas {r['prob_serangan']:.1%}). "
                     f"Kategori paling mungkin: **{r['kategori_prediksi']}**")
        else:
            st.success(f"**Normal** (probabilitas serangan {r['prob_serangan']:.1%})")
        # [Modifikasi 4] Penjelasan kategori hasil prediksi
        kartu_kategori(r["kategori_prediksi"])
        # [Modifikasi 5] Simpan ke riwayat
        st.session_state["riwayat"].append({
            "waktu": datetime.now().strftime("%H:%M:%S"),
            "contoh_awal": pilihan,
            "proto": row["proto"], "service": row["service"], "state": row["state"],
            "prob_serangan": float(r["prob_serangan"]),
            "prediksi": r["prediksi"],
            "kategori_prediksi": r["kategori_prediksi"],
            "threshold": threshold,
        })

    if st.session_state["riwayat"]:
        st.subheader("Riwayat prediksi")
        riwayat_df = pd.DataFrame(st.session_state["riwayat"][::-1])
        st.dataframe(riwayat_df, width="stretch", hide_index=True)
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button("⬇️ Unduh riwayat", riwayat_df.to_csv(index=False).encode(),
                           "riwayat_prediksi.csv", "text/csv")
        if b2.button("🗑️ Hapus riwayat"):
            st.session_state["riwayat"] = []
            st.rerun()

# --- Tab 3: Kamus serangan ---
with tab_kamus:
    st.write("Arti tiap kategori pada dataset UNSW-NB15 dan langkah awal yang biasa dilakukan administrator jaringan.")
    kamus_df = pd.DataFrame(
        [(k, a, s) for k, (a, s) in KAMUS_SERANGAN.items()],
        columns=["Kategori", "Penjelasan", "Saran tindak lanjut"])
    st.dataframe(kamus_df, width="stretch", hide_index=True)
    st.caption("Sumber definisi: Moustafa & Slay (2015), UNSW-NB15, MilCIS. Saran bersifat umum, bukan pengganti analisis insiden.")

# --- Tab 4: Info model ---
with tab_info:
    st.subheader("Fitur terpenting (model biner)")
    imp = pd.Series(meta["importance_top15"]).rename_axis("fitur").reset_index(name="importance")
    st.bar_chart(imp, x="fitur", y="importance", horizontal=True, sort="-importance")
    st.subheader("Metrik pada data test")
    st.table(pd.DataFrame({
        "Model biner": {k: round(v, 4) for k, v in meta["metrics_binary"].items()},
    }))
    st.table(pd.DataFrame({
        "Model multikelas": {k: round(v, 4) for k, v in meta["metrics_multiclass"].items()},
    }))
    st.caption(f"Dilatih dengan Python {meta['versions']['python']}, "
               f"scikit-learn {meta['versions']['scikit-learn']}, numpy {meta['versions']['numpy']}.")
