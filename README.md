# Final Project Data Mining - Credit Card Fraud Detection

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk mendeteksi transaksi penipuan kartu kredit menggunakan teknik machine learning dengan penanganan data tidak seimbang (imbalanced data). Dataset yang digunakan adalah **Credit Card Fraud Detection Dataset**.

## 🎯 Tujuan

1. Membangun model klasifikasi untuk mendeteksi transaksi fraud
2. Mengatasi masalah ketidakseimbangan data menggunakan berbagai teknik
3. Membandingkan performa model dengan berbagai skenario augmentasi data

---

## 📁 Struktur File

```
FP/
├── FinalProjectDatmin.ipynb    # Notebook utama
├── creditcard.csv              # Dataset (perlu diunduh terpisah)
├── README.md                   # Dokumentasi proyek
└── .gitignore                  # File gitignore
```

---

## 🔧 Dependencies

```python
numpy
pandas
matplotlib
scikit-learn
xgboost
imbalanced-learn
shap
lime
sdv (untuk CTGAN)
```

### Instalasi Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn shap lime sdv
```

---

## 📊 Alur Proses (Pipeline)

### 1. **Import Library dan Setup**
- Import semua library yang diperlukan
- Set random state untuk reproducibility (`RANDOM_STATE = 42`)

### 2. **Load Data dan Preprocessing**
- Load dataset `creditcard.csv`
- Pisahkan fitur (X) dan target (y) - kolom `Class`
- Split data menjadi train (80%) dan test (20%) dengan stratified sampling
- Menggunakan `RobustScaler` untuk normalisasi fitur

### 3. **Evaluasi Metrics**
Metrics yang digunakan untuk evaluasi model:
- **F1-Score**: Harmonic mean dari precision dan recall
- **AUPRC (Area Under Precision-Recall Curve)**: Lebih cocok untuk imbalanced data dibanding AUC-ROC
- **Confusion Matrix**: Visualisasi prediksi vs aktual

---

## 🧪 Skenario Eksperimen

### **Skenario 1: Baseline (Tanpa Oversampling)**

Model dilatih langsung pada data asli tanpa penanganan ketidakseimbangan.

**Model yang digunakan:**
- Random Forest Classifier
- XGBoost Classifier

**Pipeline:**
```
RobustScaler → Classifier
```

---

### **Skenario 2: SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE adalah teknik oversampling yang menghasilkan sampel sintetis untuk kelas minoritas dengan cara menginterpolasi antara sampel-sampel yang ada.

**Cara Kerja SMOTE:**
1. Memilih sampel dari kelas minoritas
2. Mencari k-nearest neighbors dari sampel tersebut
3. Membuat sampel sintetis baru di sepanjang garis yang menghubungkan sampel asli dengan tetangganya

**Konfigurasi:**
```python
SMOTE(
    random_state=42,
    k_neighbors=3,
    sampling_strategy=0.1  # Fraud menjadi 10% dari normal
)
```

**Pipeline:**
```
RobustScaler → SMOTE → Classifier
```

**Referensi:**
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

---

### **Skenario 3: CTGAN (Conditional Tabular Generative Adversarial Network)**

CTGAN adalah teknik generatif berbasis deep learning yang dirancang khusus untuk menghasilkan data tabular sintetis berkualitas tinggi.

**Keunggulan CTGAN:**
- Menangani data campuran (numerik dan kategorikal)
- Menghasilkan data sintetis yang mempertahankan distribusi statistik data asli
- Mengatasi masalah mode collapse pada data tabular
- Lebih realistis dibanding interpolasi sederhana

**Konfigurasi:**
```python
CTGANSynthesizer(
    metadata,
    epochs=10,
    batch_size=64,
    pac=1,
    enforce_min_max_values=True,
    enforce_rounding=False
)
```

**Proses:**
1. Subset data fraud untuk training CTGAN (max 300 sampel)
2. Fit CTGAN pada data fraud only
3. Generate sampel sintetis fraud
4. Gabungkan dengan data original untuk training

**Pipeline:**
```
Data Original + Data Sintetis CTGAN → RobustScaler → Classifier
```

**Referensi:**
> Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling Tabular Data using Conditional GAN. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

---

## 🤖 Model Machine Learning

### Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=120,
    max_depth=18,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
```

### XGBoost Classifier
```python
XGBClassifier(
    n_estimators=220,
    max_depth=4,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist"
)
```

---

## 📈 Hasil dan Evaluasi

### Metrics Evaluasi
| Model | Skenario | F1-Score | AUPRC |
|-------|----------|----------|-------|
| RF_Baseline | Baseline | - | - |
| XGB_Baseline | Baseline | - | - |
| RF_SMOTE | SMOTE(0.1) | - | - |
| XGB_SMOTE | SMOTE(0.1) | - | - |
| RF_CTGAN | CTGAN | - | - |
| XGB_CTGAN | CTGAN | - | - |

*Catatan: Jalankan notebook untuk mendapatkan hasil aktual*

### Visualisasi
1. **Confusion Matrix**: Untuk setiap kombinasi model dan skenario
2. **Precision-Recall Curve**: Perbandingan semua model dalam satu plot

---

## 📝 Analisis Delta (Perbandingan)

Notebook menyediakan fungsi `show_delta()` untuk membandingkan performa:
- Baseline vs CTGAN
- Peningkatan/penurunan F1 dan AUPRC

**Kesimpulan:**
- Jika AUPRC meningkat → Teknik augmentasi efektif
- Jika AUPRC menurun → Teknik augmentasi belum efektif

---

## 🚀 Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/kamilramadhan/fp-datmin.git
   cd fp-datmin
   ```

2. **Download dataset**
   - Download `creditcard.csv` dari [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Letakkan di folder yang sama dengan notebook

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan notebook**
   - Buka `FinalProjectDatmin.ipynb` di Jupyter Notebook atau VS Code
   - Run all cells

---

## 📚 Referensi

1. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.

2. Xu, L., et al. (2019). Modeling Tabular Data using Conditional GAN. *NeurIPS*, 32.

3. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 1-48.

4. Dal Pozzolo, A., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE SSCI*.

---

## 👥 Kelompok 4

Final Project Data Mining - Semester 5

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis.
