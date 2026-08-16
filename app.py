import streamlit as st
import pickle
import pandas as pd
import time

# Load model Iris (pipeline: scaler + SVM terbaik)
pklname = "generate_iris.pkl"
with open(pklname, 'rb') as file:
    pick = pickle.load(file)

# Load model Heart Disease (GridSearchCV, best_estimator_ = RandomForestClassifier)
pklname_hd = "generate_heart_disease.pkl"
with open(pklname_hd, 'rb') as file:
    pick_hd = pickle.load(file)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["About Me", "Predict Bunga Iris", "Predict Penyakit Jantung"])

# ------------------ HALAMAN: ABOUT ME ------------------
if menu == "About Me":
    st.markdown(
        """
        <div style="padding:22px 26px;border-radius:14px;
                    background:linear-gradient(135deg,#6C63FF 0%,#B983FF 50%,#E0245E 100%);
                    color:white;margin-bottom:20px;">
            <h1 style="margin:0;color:white;">👋 About Me</h1>
            <p style="margin:6px 0 0 0;font-size:16px;opacity:0.95;">
                Data Enthusiast — belajar & membangun proyek Machine Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Halo! Saya **Khoirul Surya Nada**, seorang *Data Enthusiast* yang senang belajar "
             "dan membangun proyek seputar data science serta machine learning.")

    st.markdown("🔗 **LinkedIn:** [linkedin.com/in/khoirul-surya-nada-118319264]"
                "(https://www.linkedin.com/in/khoirul-surya-nada-118319264)")
    st.markdown("📧 **Email:** khoirul114499@gmail.com")

    st.divider()
    st.subheader("🧭 Tentang Aplikasi Ini")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="padding:16px;border-radius:12px;background:#f3e8ff;border:1px solid #d8b4fe;">
                <h4 style="margin:0 0 6px 0;color:#3b0764;">🌸 Predict Bunga Iris</h4>
                <p style="margin:0;font-size:14px;color:#3b0764;">
                Memprediksi spesies bunga Iris (Setosa, Versicolor, Virginica) berdasarkan
                ukuran sepal dan petal, menggunakan model SVM.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="padding:16px;border-radius:12px;background:#ffe4e6;border:1px solid #fda4af;">
                <h4 style="margin:0 0 6px 0;color:#881337;">❤️ Predict Penyakit Jantung</h4>
                <p style="margin:0;font-size:14px;color:#881337;">
                Memprediksi risiko penyakit jantung berdasarkan data kondisi pasien,
                menggunakan model Random Forest.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------ HALAMAN: PREDICT IRIS ------------------
elif menu == "Predict Bunga Iris":
    st.markdown(
        """
        <div style="padding:20px 26px;border-radius:14px;
                    background:linear-gradient(135deg,#8E7CF0 0%,#B983FF 100%);
                    color:white;margin-bottom:18px;">
            <h1 style="margin:0;color:white;">🌸 Prediksi Spesies Bunga Iris</h1>
            <p style="margin:6px 0 0 0;font-size:15px;opacity:0.95;">
                Setosa • Versicolor • Virginica
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("Geser slider atau upload file di sidebar untuk mengatur ukuran bunga, lalu klik tombol **Predict**.")
    st.image("3 iris.png", width=1500)

    with st.expander("🌿 Penjelasan Bagian Bunga & Cara Mengukurnya (klik untuk membuka)"):
        st.markdown("""
Bunga Iris memiliki dua bagian utama yang diukur pada aplikasi ini: **Sepal** dan **Petal**. Lihat gambar panduan di sidebar untuk contoh visualnya.

**Sepal (Kelopak Luar)**
Bagian daun kecil berwarna hijau/keunguan yang berada di lapisan **terluar** bunga, biasanya melindungi kuncup sebelum bunga mekar. Pada Iris, sepal justru tampak lebar dan berwarna mencolok, sering disangka sebagai mahkota bunga.
- *Panjang Sepal* — diukur dari pangkal sepal (dekat batang) hingga ujung terluarnya.
- *Lebar Sepal* — diukur pada bagian terlebar dari sepal, tegak lurus dengan arah panjangnya.

**Petal (Mahkota Bunga)**
Bagian mahkota bunga yang berada **di dalam** sepal, biasanya lebih kecil, tegak, dan berwarna cerah — ini bagian yang paling sering dianggap "kelopak bunga" oleh orang awam.
- *Panjang Petal* — diukur dari pangkal petal hingga ujung terluarnya.
- *Lebar Petal* — diukur pada bagian terlebar dari petal, tegak lurus dengan arah panjangnya.

---
💡 **Tips:** Jika Anda mengukur bunga Iris asli, gunakan penggaris kecil (dalam satuan cm) dan ukur pada bagian terlebar/terpanjang tiap bagian. Jika hanya mengira-ngira, sesuaikan dengan gambar panduan di sidebar sebagai referensi bentuk umum tiap bagian.
        """)
    st.sidebar.image("iris_guide.png", width=1000)

    # Pisahkan opsi input di sidebar: Slider atau Upload File
    input_mode = st.sidebar.radio("Metode Input", ["Manual", "Upload File"])

    if input_mode == "Manual":
        # Semua input & tombol predict ditaruh di sidebar
        st.sidebar.subheader("Input Ukuran Bunga Iris")
        sepal_length = st.sidebar.slider("Panjang Sepal (cm)", min_value=4.0, max_value=10.0, value=5.8, step=0.1)
        sepal_width = st.sidebar.slider("Lebar Sepal (cm)", min_value=2.0, max_value=10.5, value=3.0, step=0.1)
        petal_length = st.sidebar.slider("Panjang Petal (cm)", min_value=1.0, max_value=10.0, value=3.7, step=0.1)
        petal_width = st.sidebar.slider("Lebar Petal (cm)", min_value=0.1, max_value=10.5, value=1.2, step=0.1)

        predict_button = st.sidebar.button("Predict")

        if predict_button:
            data = {
                'Panjang Sepal': sepal_length,
                'Lebar Sepal': sepal_width,
                'Panjang Petal': petal_length,
                'Lebar Petal': petal_width
            }
            features = pd.DataFrame(data, index=[0])

            # Tampilkan tabel data hasil inputan di halaman utama
            st.subheader("Data Inputan")
            st.table(features)

            # Animasi loading untuk menandakan model sedang bekerja
            with st.spinner("Model sedang memproses prediksi..."):
                time.sleep(1.5)
                prediction = pick.predict(features)

            result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]

            st.subheader("Hasil Prediksi")
            st.success(f"Spesies bunga yang terprediksi adalah: **{result[0]}**")

    elif input_mode == "Upload File":
        st.sidebar.subheader("Upload Data")

        with open("tamplate_iris.xlsx", "rb") as f:
            st.sidebar.download_button(
                label="Download Template (tamplate_iris.xlsx)",
                data=f,
                file_name="tamplate_iris.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        uploaded_file = st.sidebar.file_uploader("Upload file (.xlsx)", type=["xlsx"])
        predict_button = st.sidebar.button("Predict")

        if uploaded_file is not None:
            features = pd.read_excel(uploaded_file)

            st.subheader("Data Inputan")
            st.table(features)

            if predict_button:
                with st.spinner("Model sedang memproses prediksi..."):
                    time.sleep(1.5)
                    prediction = pick.predict(features)

                result = ['Iris-setosa' if p == 0 else ('Iris-versicolor' if p == 1 else 'Iris-virginica') for p in prediction]

                st.subheader("Hasil Prediksi")
                features_result = features.copy()
                features_result["Hasil Prediksi"] = result
                st.table(features_result)


# ------------------ HALAMAN: PREDICT HEART DISEASE ------------------
elif menu == "Predict Penyakit Jantung":
    st.markdown(
        """
        <div style="padding:20px 26px;border-radius:14px;
                    background:linear-gradient(135deg,#B91C1C 0%,#E11D48 100%);
                    color:white;margin-bottom:18px;">
            <h1 style="margin:0;color:white;">❤️ Prediksi Risiko Penyakit Jantung</h1>
            <p style="margin:6px 0 0 0;font-size:15px;opacity:0.95;">
                Berdasarkan data kondisi pasien
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("Isi data di sidebar atau upload file untuk mengatur kondisi pasien, lalu klik tombol **Predict**.")

    st.warning(
        "⚠️ **Disclaimer:** Hasil pada aplikasi ini hanyalah prediksi berbasis model machine learning "
        "dan **bukan diagnosis medis**. Apapun hasilnya, disarankan untuk tetap memeriksakan diri "
        "lebih lanjut kepada dokter atau tenaga medis profesional."
    )

    with st.expander("📋 Penjelasan Fitur & Cara Mendapatkan Angkanya (klik untuk membuka)"):
        st.markdown("""
**Usia**
Isi sesuai usia Anda saat ini (dalam tahun).

**Jenis Kelamin**
Pilih sesuai jenis kelamin Anda.

**Tipe Nyeri Dada**
Pilih jenis nyeri dada yang paling sering Anda rasakan:
- *Angina Umum/Khas* — nyeri dada yang muncul saat beraktivitas berat dan mereda saat istirahat.
- *Angina Tidak Umum* — nyeri dada dengan pola yang tidak lazim/tidak khas.
- *Nyeri Bukan dari Jantung* — nyeri dada yang diketahui bukan berasal dari jantung (misal otot, lambung).
- *Tanpa Gejala Nyeri* — tidak pernah merasakan nyeri dada sama sekali.

Jika ragu, ini bisa dilihat dari catatan rekam medis atau hasil konsultasi dokter sebelumnya.

**Detak Jantung Maksimum**
Angka detak jantung tertinggi yang pernah tercatat saat Anda beraktivitas fisik berat (misal saat treadmill test/tes jantung dengan olahraga). Bisa dilihat dari hasil pemeriksaan EKG/tes stres jantung, atau perkiraan dari alat pengukur detak jantung (smartwatch, dll) saat berolahraga maksimal.

**Nyeri Dada Saat Beraktivitas Fisik**
Pilih "Ya" jika Anda pernah merasakan nyeri dada saat berolahraga/beraktivitas berat, atau "Tidak" jika belum pernah.

**Tingkat Kelelahan Jantung Setelah Aktivitas Fisik (ST Depression)**
Angka ini berasal dari hasil rekam jantung (EKG) yang mengukur seberapa besar penurunan sinyal listrik jantung setelah beraktivitas fisik dibanding saat istirahat. Angka ini biasanya tertera pada laporan hasil EKG/tes stres jantung dari dokter. Semakin besar angkanya, semakin besar indikasi kelelahan jantung.

**Pola Perubahan Aktivitas Jantung Saat Berolahraga**
Juga berasal dari hasil EKG — menunjukkan pola garis sinyal jantung (segmen ST) saat beraktivitas: menurun, datar, atau meningkat. Informasi ini ada pada laporan hasil EKG.

**Jumlah Pembuluh Darah Utama yang Tersumbat**
Jumlah pembuluh darah utama (0 sampai 3) yang terdeteksi menyempit/tersumbat berdasarkan hasil pemeriksaan pencitraan jantung (misal angiografi/fluoroskopi). Angka ini didapat dari hasil pemeriksaan dokter spesialis jantung.

**Kondisi Aliran Darah ke Jantung (Thalassemia)**
Hasil pemeriksaan thalium/thalassemia yang menunjukkan kondisi aliran darah ke otot jantung: Normal, Kerusakan Permanen (defect yang sudah menetap), atau Kerusakan Sementara (defect yang muncul hanya saat beraktivitas). Informasi ini berasal dari hasil tes thalium scan yang dilakukan dokter.

---
💡 **Tips:** Semua nilai di atas (kecuali usia dan jenis kelamin) idealnya diambil dari hasil pemeriksaan/rekam medis resmi, bukan perkiraan sendiri, supaya hasil prediksi lebih akurat dan relevan.
        """)

    # Urutan kolom WAJIB sama persis dengan urutan fitur saat model dilatih
    FEATURE_ORDER = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age']

    input_mode_hd = st.sidebar.radio("Metode Input", ["Manual", "Upload File"], key="hd_mode")

    # Label tampilan yang menyertakan nama fitur asli, dipakai konsisten
    # di sidebar (Input Data Pasien) maupun di tabel (Data Inputan)
    FEATURE_LABELS = {
        'age': "Usia (age)",
        'sex': "Jenis Kelamin (sex)",
        'cp': "Tipe Nyeri Dada (cp)",
        'thalach': "Detak Jantung Maksimum (thalach)",
        'exang': "Nyeri Dada Saat Beraktivitas Fisik (exang)",
        'oldpeak': "Tingkat Kelelahan Jantung Setelah Aktivitas Fisik (oldpeak)",
        'slope': "Pola Perubahan Aktivitas Jantung Saat Berolahraga (slope)",
        'ca': "Jumlah Pembuluh Darah Utama yang Tersumbat (ca)",
        'thal': "Kondisi Aliran Darah ke Jantung (thal)",
    }

    # Mapping teks kategori (dipakai untuk input manual maupun upload file)
    SEX_MAP = {"Laki-laki": 1, "Perempuan": 0}
    CP_MAP = {
        "Angina Umum / Khas (Typical Angina)": 0,
        "Angina Tidak Umum (Atypical Angina)": 1,
        "Nyeri Bukan dari Jantung (Non-anginal Pain)": 2,
        "Tanpa Gejala Nyeri (Asymptomatic)": 3
    }
    EXANG_MAP = {"Tidak": 0, "Ya": 1}
    SLOPE_MAP = {"Menurun (Downsloping)": 0, "Datar (Flat)": 1, "Meningkat (Upsloping)": 2}
    THAL_MAP = {"Normal": 1, "Kerusakan Permanen (Fixed Defect)": 2, "Kerusakan Sementara (Reversible Defect)": 3}

    # Kolom pada file template (teks kategori, urutan bebas asalkan nama kolomnya cocok)
    TEMPLATE_COLUMNS = {
        'Tipe Nyeri Dada': 'cp',
        'Detak Jantung Maksimum': 'thalach',
        'Pola Perubahan Aktivitas Jantung Saat Berolahraga': 'slope',
        'Tingkat Kelelahan Jantung Setelah Aktivitas Fisik': 'oldpeak',
        'Nyeri Dada Saat Beraktivitas Fisik': 'exang',
        'Jumlah Pembuluh Darah Utama yang Tersumbat': 'ca',
        'Kondisi Aliran Darah ke Jantung': 'thal',
        'Jenis Kelamin': 'sex',
        'Usia': 'age',
    }

    def encode_uploaded_heart_data(df_raw):
        """Konversi file upload (berisi teks kategori & angka) menjadi DataFrame
        angka yang siap dipakai model, dengan validasi & pesan error yang jelas."""
        df_raw = df_raw.rename(columns=TEMPLATE_COLUMNS)
        missing_cols = [c for c in FEATURE_ORDER if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan di file: {', '.join(missing_cols)}. "
                     f"Pastikan menggunakan template yang disediakan.")
            return None

        df_raw = df_raw.dropna(subset=FEATURE_ORDER, how='all').reset_index(drop=True)
        encoded_rows = []
        for i, row in df_raw.iterrows():
            try:
                encoded_rows.append({
                    'cp': CP_MAP[row['cp']],
                    'thalach': int(row['thalach']),
                    'slope': SLOPE_MAP[row['slope']],
                    'oldpeak': float(row['oldpeak']),
                    'exang': EXANG_MAP[row['exang']],
                    'ca': int(row['ca']),
                    'thal': THAL_MAP[row['thal']],
                    'sex': SEX_MAP[row['sex']],
                    'age': int(row['age']),
                })
            except KeyError as e:
                st.error(f"Baris {i + 2}: nilai {e} tidak dikenali. "
                         f"Pastikan mengisi sesuai pilihan dropdown pada template.")
                return None
            except (ValueError, TypeError):
                st.error(f"Baris {i + 2}: ada nilai angka yang tidak valid (kosong atau bukan angka).")
                return None

        return pd.DataFrame(encoded_rows)[FEATURE_ORDER]

    if input_mode_hd == "Manual":
        st.sidebar.subheader("Input Data Pasien")

        age = st.sidebar.number_input(FEATURE_LABELS['age'], min_value=20, max_value=80, value=45, step=1)

        sex_label = st.sidebar.radio(FEATURE_LABELS['sex'], ["Laki-laki", "Perempuan"], horizontal=True)

        cp_label = st.sidebar.selectbox(
            FEATURE_LABELS['cp'],
            [
                "Angina Umum / Khas (Typical Angina)",
                "Angina Tidak Umum (Atypical Angina)",
                "Nyeri Bukan dari Jantung (Non-anginal Pain)",
                "Tanpa Gejala Nyeri (Asymptomatic)"
            ]
        )

        thalach = st.sidebar.number_input(FEATURE_LABELS['thalach'], min_value=70, max_value=210, value=150, step=1)

        exang_label = st.sidebar.radio(FEATURE_LABELS['exang'], ["Tidak", "Ya"], horizontal=True)

        oldpeak = st.sidebar.slider(
            FEATURE_LABELS['oldpeak'],
            min_value=0.0, max_value=6.5, value=1.0, step=0.1
        )

        slope_label = st.sidebar.selectbox(
            FEATURE_LABELS['slope'],
            ["Menurun (Downsloping)", "Datar (Flat)", "Meningkat (Upsloping)"]
        )

        ca = st.sidebar.selectbox(FEATURE_LABELS['ca'], [0, 1, 2, 3])

        thal_label = st.sidebar.selectbox(
            FEATURE_LABELS['thal'],
            ["Normal", "Kerusakan Permanen (Fixed Defect)", "Kerusakan Sementara (Reversible Defect)"]
        )

        predict_button_hd = st.sidebar.button("Predict", key="hd_predict_manual")

        if predict_button_hd:
            # Encoding sesuai mapping yang dipakai saat training
            sex = 1 if sex_label == "Laki-laki" else 0
            cp_map = {
                "Angina Umum / Khas (Typical Angina)": 0,
                "Angina Tidak Umum (Atypical Angina)": 1,
                "Nyeri Bukan dari Jantung (Non-anginal Pain)": 2,
                "Tanpa Gejala Nyeri (Asymptomatic)": 3
            }
            cp = cp_map[cp_label]
            exang = 1 if exang_label == "Ya" else 0
            slope_map = {"Menurun (Downsloping)": 0, "Datar (Flat)": 1, "Meningkat (Upsloping)": 2}
            slope = slope_map[slope_label]
            thal_map = {"Normal": 1, "Kerusakan Permanen (Fixed Defect)": 2, "Kerusakan Sementara (Reversible Defect)": 3}
            thal = thal_map[thal_label]

            data = {
                'cp': cp, 'thalach': thalach, 'slope': slope, 'oldpeak': oldpeak,
                'exang': exang, 'ca': ca, 'thal': thal, 'sex': sex, 'age': age
            }
            features = pd.DataFrame(data, index=[0])[FEATURE_ORDER]

            # Tabel ditampilkan vertikal (Fitur | Nilai) supaya ringkas, tidak melebar ke bawah
            st.subheader("Data Inputan")
            display_df = features.rename(columns=FEATURE_LABELS).T.reset_index()
            display_df.columns = ["Fitur", "Nilai"]
            st.table(display_df)

            with st.spinner("Model sedang memproses prediksi..."):
                time.sleep(1.5)
                prediction = pick_hd.predict(features)

            result = "Disease (Berisiko Penyakit Jantung)" if prediction[0] == 1 else "No Disease (Tidak Berisiko)"

            st.subheader("Hasil Prediksi")
            if prediction[0] == 1:
                st.error(f"Hasil: **{result}**")
            else:
                st.success(f"Hasil: **{result}**")

            st.info(
                "ℹ️ Hasil ini hanya prediksi model dan tidak menggantikan pemeriksaan medis. "
                "Tetap konsultasikan ke dokter untuk kepastian diagnosis."
            )

    elif input_mode_hd == "Upload File":
        st.sidebar.subheader("Upload Data")

        with open("tamplate_heart_disease.xlsx", "rb") as f:
            st.sidebar.download_button(
                label="Download Template (tamplate_heart_disease.xlsx)",
                data=f,
                file_name="tamplate_heart_disease.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="hd_download"
            )

        uploaded_file_hd = st.sidebar.file_uploader("Upload file (.xlsx)", type=["xlsx"], key="hd_upload")
        predict_button_hd_upload = st.sidebar.button("Predict", key="hd_predict_upload")

        if uploaded_file_hd is not None:
            features_raw = pd.read_excel(uploaded_file_hd)
            features = encode_uploaded_heart_data(features_raw)

            if features is not None:
                st.subheader("Data Inputan")
                st.dataframe(features_raw, use_container_width=True)

                if predict_button_hd_upload:
                    with st.spinner("Model sedang memproses prediksi..."):
                        time.sleep(1.5)
                        prediction = pick_hd.predict(features)

                    result = ["Disease" if p == 1 else "No Disease" for p in prediction]

                    st.subheader("Hasil Prediksi")
                    features_result = features_raw.copy()
                    features_result["Hasil Prediksi"] = result
                    st.dataframe(features_result, use_container_width=True)

                    st.info(
                        "ℹ️ Hasil ini hanya prediksi model dan tidak menggantikan pemeriksaan medis. "
                        "Tetap konsultasikan ke dokter untuk kepastian diagnosis."
                    )