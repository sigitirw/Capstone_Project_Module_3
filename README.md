# Capstone_Project_Module_3
Capstone Project Module 3 sebagai tugas dari bootcamp data science Purwadhika School
Repositori ini berisi analisis lengkap, eksplorasi data, pemodelan machine learning, serta deployment sederhana untuk memprediksi customer churn pada perusahaan telekomunikasi. Notebook utama:
➡️ Capstone_Project_Module_3_s.ipynb
📝 Deskripsi Project
Tujuan project ini adalah membangun model klasifikasi yang mampu memprediksi apakah seorang pelanggan berpotensi melakukan churn (berhenti berlangganan), sehingga perusahaan dapat mengambil langkah mitigasi seperti penawaran khusus atau retensi.
Proses yang dilakukan meliputi:
Exploratory Data Analysis (EDA)
Data cleaning & preprocessing
Feature engineering
Benchmarking berbagai model ML
Evaluasi berdasarkan metrik bisnis (recall, precision, ROC-AUC)
Threshold tuning untuk memaksimalkan recall
Simulasi biaya (business cost analysis)
Pembuatan model akhir & deployment sederhana
📁 Struktur Repository
├── Capstone_Project_Module_3_s.ipynb   # Notebook utama analisis dan modeling
├── data_telco_customer_churn.csv       # Dataset (jika disertakan)
├── telcoChurn.pkl                      # Model final dalam format pickle (opsional)
├── app.py                              # Aplikasi Streamlit (jika digunakan)
├── predictions_log.csv                 # Log prediksi (opsional)
└── README.md                           # Dokumentasi proyek
🔍 Highlights Analisis
1. Exploratory Data Analysis (EDA)
EDA mencakup:
Distribusi variabel kategorikal & numerik
Churn rate secara keseluruhan
Korelasi antar numeric features
Analisis fitur penting seperti:
Contract
Tenure
MonthlyCharges
Insight pola pelanggan churn vs loyal
2. Data Preprocessing
Beberapa langkah kunci:
Penanganan missing values
Encoding variabel kategorikal
Normalisasi/standarisasi
Train–test split
🤖 Machine Learning Modeling
Model yang dievaluasi termasuk:
Logistic Regression
Random Forest
XGBoost
KNN
SVM
Metrik utama yang digunakan bergantung pada tujuan bisnis:
📌 Recall diprioritaskan agar sebanyak mungkin churn terdeteksi.
Evaluasi mencakup:
Confusion Matrix
ROC-AUC
Precision–Recall Curve
Threshold tuning (custom)
Cost/benefit analysis FP vs FN
🚀 Deployment (Opsional)
Jika menggunakan file app.py, aplikasi Streamlit sudah mendukung:
EDA interaktif
Prediksi single customer
Batch prediction (CSV upload)
Logging otomatis prediksi
Visualisasi probabilitas churn
Jalankan aplikasi dengan:
streamlit run app.py
📦 Instalasi & Dependencies
Install library yang dibutuhkan:
pip install -r requirements.txt
Contoh requirements.txt:
streamlit
pandas
numpy
scikit-learn
joblib
plotly
matplotlib
xgboost
📈 Hasil Utama Model
(Opsional — tambahkan angka dari notebook)
ROC-AUC: ....
Recall @ optimal threshold: ....
Akurasi: ....
Business cost saving: ....
🧠 Insight Bisnis
Beberapa rekomendasi dari analisis:
Fokus retensi pada pelanggan dengan kontrak bulanan (Month-to-month)
Pelanggan dengan tenure rendah memiliki risiko churn lebih tinggi
Add-on security & tech support memiliki dampak signifikan
Target intervensi sebaiknya diarahkan ke segmen dengan prob. churn tinggi tetapi biaya intervensi rendah
