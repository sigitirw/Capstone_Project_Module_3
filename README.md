# 📊 Telco Customer Churn Prediction
**Capstone Project — Module 3**  
**By: Sigit Irwanto**

---

## 🔎 Ringkasan Project
Project ini membangun pipeline untuk memprediksi **customer churn** pada perusahaan telekomunikasi. Meliputi: EDA, preprocessing reproducible, benchmarking model, hyperparameter tuning (fokus **recall**), threshold tuning dengan simulasi biaya bisnis, serta deployment ringan (Streamlit).

Notebook utama: `Capstone_Project_Module_3_s.ipynb`  
Opsional: Streamlit app `app.py` dan model pickled `telcoChurn.pkl`.

---

## 🗂️ Struktur Repository (disarankan)

📁 Struktur Repository
├── Capstone_Project_Module_3_s.ipynb   # Notebook utama analisis dan modeling
├── data_telco_customer_churn.csv       # Dataset (jika disertakan)
├── telcoChurn.pkl                      # Model final dalam format pickle (opsional)
├── app.py                              # Aplikasi Streamlit (jika digunakan)
├── predictions_log.csv                 # Log prediksi (opsional)
└── README.md                           # Dokumentasi proyek


---

## 📄 Deskripsi Data
File: `data_telco_customer_churn.csv`  
Beberapa fitur penting:
- `Dependents`, `tenure`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `Contract`, `PaperlessBilling`, `MonthlyCharges`  
- Target: `Churn` (Yes / No)

(Data dictionary lengkap dan handling missing ada di notebook.)

---

## 🧭 Tujuan Bisnis & Metrik
**Tujuan:** Menangkap sebanyak mungkin churners agar tim retensi bisa melakukan intervensi.  
**Metrik utama:** **Recall** (mengutamakan minim False Negative).  
Metrik tambahan: Precision, F1, ROC-AUC, confusion matrix, PR-AUC, dan simulasi biaya FP vs FN.

---

## 📊 Highlights Analisis & Insight
- Kontrak **Month-to-month** memiliki churn tertinggi.  
- Layanan **Fiber optic** cenderung punya churn rate lebih tinggi.  
- Ketiadaan add-on (TechSupport, OnlineSecurity) berkaitan dengan meningkatnya churn.  
- Tenure rendah → risiko churn lebih besar.  
- MonthlyCharges berasosiasi positif dengan churn.

(Rekomendasi bisnis dan visualisasi detail tersedia di notebook.)

---

## ⚙️ Preprocessing & Modelling
- Missing value handling (termasuk konversi `TotalCharges` jika perlu)
- Encoding kategorikal + scaling numeric di pipeline scikit-learn
- Models evaluated: Logistic Regression, Random Forest, XGBoost, KNN, SVM
- Hyperparameter tuning: GridSearchCV (objective: maximize recall)
- Final model: **Tuned Logistic Regression** (contoh hyperparams ada di notebook)

## 📈 Model Performance (Dari Notebook)

### 🔹 Benchmark ROC-AUC (Cross-Validation)
Hasil perbandingan beberapa model (mean ROC-AUC):

| Model                   | ROC-AUC |
|-------------------------|---------|
| **Logistic Regression** | **83.87%** |
| SVM                     | 82.05% |
| Gaussian Naive Bayes    | 81.58% |
| XGBoost                 | 80.79% |
| Random Forest           | 79.67% |
| KNN                     | 77.80% |
| Decision Tree           | 64.68% |

> Logistic Regression menjadi kandidat model terbaik untuk tuning.

---

## 🔧 Logistic Regression — Hyperparameter Tuning (GridSearchCV)

### ⭐ Best CV Recall:
**82.90%**

### ⭐ Best Params:
```python
{
  'clf__C': 0.001,
  'clf__class_weight': 'balanced',
  'clf__max_iter': 200,
  'clf__penalty': 'l2',
  'clf__solver': 'liblinear'
}
## 📊 Threshold Analysis & Classification Metrics

### **Default Threshold (0.5) — Tuned Model**
| Metric       | Value    |
|--------------|----------|
| **Recall**   | 0.8517   |
| **Precision**| 0.5056   |
| **F1-score** | 0.6346   |
| **ROC-AUC**  | 0.8538   |
| **PR-AUC**   | 0.6785   |

### **Confusion Matrix**
|        | Predicted: No | Predicted: Yes |
|--------|---------------|----------------|
| **Actual: No**  | TN = 504      | FP = 219        |
| **Actual: Yes** | FN = 39       | TP = 224        |

---

## 💰 Business Cost Simulation
Simulasi biaya dihitung berdasarkan dua komponen utama:
- **Biaya kehilangan pelanggan (FN)**  
- **Biaya kampanye/intervensi (FP)**  

### 🔹 **Default Model — Threshold Optimal (0.31)**
| Metric       | Value       |
|--------------|-------------|
| **Total Cost** | **22,260,000** |
| FP           | 306         |
| FN           | 13          |

### 🔹 **Tuned Model — Threshold Optimal (0.43)**
| Metric       | Value       |
|--------------|-------------|
| **Total Cost** | **23,700,000** |
| FP           | 320         |
| FN           | 15          |

> **Default model pada threshold 0.31 menghasilkan biaya total paling rendah**, sehingga merupakan pilihan optimal dari perspektif bisnis.

---

## 📌 Final Model Summary
- **Model terbaik (bisnis):** Logistic Regression (default) + threshold 0.31  
- **Model terbaik (statistik):** Logistic Regression (tuned)  
- **Model chosen for deployment:** dapat disesuaikan dengan prioritas bisnis (recall tinggi vs FP lebih rendah)  
- **Recall tetap menjadi metrik utama**, karena False Negative jauh lebih mahal dibanding False Positive.

---

## 🔥 Key Business Insights (From Model)
- Pelanggan **Month-to-month** adalah penyumbang churn terbesar.  
- Churn rate meningkat pada pelanggan **Fiber optic**.  
- Pelanggan tanpa **TechSupport** dan **OnlineSecurity** lebih rentan churn.  
- **Tenure rendah** (pelanggan baru) memiliki churn probability jauh lebih tinggi.  
- **MonthlyCharges** yang tinggi berasosiasi positif dengan churn.  
