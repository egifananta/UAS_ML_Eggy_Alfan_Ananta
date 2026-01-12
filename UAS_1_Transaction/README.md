## 1. Tujuan Repositori
Repositori ini dibuat untuk memenuhi UAS Mata Kuliah Machine Learning dengan membangun sistem pendeteksian transaksi penipuan (fraud detection) secara end-to-end, mulai dari loading data, preprocessing, penanganan imbalance, training model Deep Learning, hingga pembuatan file submission untuk data uji.

**Target utama:**
- Membangun pipeline end-to-end yang efisien dan ramah memori di Google Colab (RAM 12GB).
- Menghasilkan model dengan metrik utama ROC-AUC dan F1-Score yang layak untuk data sangat tidak seimbang.
- Menerapkan *feature engineering*, balancing (SMOTE + class weight + focal loss), dan evaluasi yang jelas.
- Menghasilkan file prediksi untuk *test set* (`submission.csv`).

---

## 2. Gambaran Singkat Proyek
Dataset transaksi online memiliki label biner **isFraud** (0 = normal, 1 = fraud) dengan fraud rate sekitar **3.5%** (sangat imbalanced).

Dari ratusan fitur asli, program ini hanya memuat kolom-kolom penting (`TransactionDT`, `TransactionAmt`, `ProductCD`, `card1–6`, `addr1–2`, C-features, D-features, M-features, beberapa V-features, email domain, dll), lalu dilakukan:

### A. Optimisasi Memori
- Downcasting tipe data numerik.
- Hanya me-load kolom penting sehingga penggunaan memori berkurang >30%.

### B. Preprocessing & Feature Engineering
- `TransactionAmt_log`, `TransactionAmt_decimal`.
- Fitur waktu: `TransactionDT_hour`, `TransactionDT_dayofweek`.
- Agregasi V, C, dan D features: `V_sum`, `V_mean`, `V_std`, `C_sum`, `C_mean`, `D_sum`, `D_mean`.
- Indicator email: `P_emaildomain_isNull`, `R_emaildomain_isNull`.
- One-hot encoding `ProductCD` dan label encoding untuk fitur kategorikal lain.
- Handling missing value dengan nilai `-999` dan drop kolom dengan missing >85%.

### C. Handling Imbalance
- **SMOTE** (`sampling_strategy=0.35`) untuk menambah data kelas fraud.
- **Class weight** (kelas fraud dibobot ±2.5x lebih besar).
- **Focal Loss** saat training untuk fokus pada *hard examples* kelas minoritas.

### D. Scaling
- Menggunakan **RobustScaler** agar lebih tahan terhadap outlier.

---

## 3. Model yang Digunakan & Hasil Evaluasi

### Model Utama: Deep Neural Network (Efficient DNN)
Model utama adalah Neural Network bertingkat dengan arsitektur:

`input → 256 → 128 → 64 → 32 → 1`

dilengkapi dengan **Batch Normalization**, **Dropout**, **L2 regularization**, dan **GaussianNoise** pada input.

**Konfigurasi penting:**
- Optimizer: **Adam** (`lr=0.001`, `clipnorm=1.0`).
- Loss: **Focal Loss** (α = 0.25, γ = 2.0).
- Callback: **EarlyStopping**, **ReduceLROnPlateau**, dan **ModelCheckpoint**.
- Total parameter: sekitar **67 ribu**.

### Hasil Validasi (Validation Set)
Menggunakan threshold optimal dari Precision–Recall curve (≈ **0.9394**), diperoleh:

| Metrik    | Skor   |
|-----------|--------|
| ROC-AUC   | 0.7626 |
| Accuracy  | 0.8674 |
| F1-Score  | 0.1853 |
| Precision | 0.1180 |
| Recall    | 0.4309 |

### Confusion Matrix  
(kelas 0 = normal, kelas 1 = fraud)

- TN = 100,663  
- FP = 13,312  
- FN = 2,352  
- TP = 1,781  

Model **cukup baik** membedakan transaksi normal dan fraud (ROC-AUC > 0.76) dengan *recall* fraud ≈ 43%, namun *precision* masih relatif rendah (banyak *false alarm*), yang umum pada kasus *fraud detection* dengan data sangat tidak seimbang.

---

## 4. Alur Program & Cara Menggunakan

### A. Lokasi Dataset & Output
Struktur folder di Google Drive (disesuaikan dengan tugas):

```text
Drive Saya/
├── ML/
│   ├── train_transaction.csv       # Dataset train asli
│   ├── test_transaction.csv        # Dataset test asli
│   └── models/
│       ├── fraud_model.h5          # Model terlatih
│       ├── scaler.pkl              # RobustScaler
│       ├── threshold.pkl           # Threshold optimal
│       ├── features.pkl            # Daftar 87 fitur yang dipakai
│       └── submission.csv          # File prediksi akhir (output)
└── Colab Notebooks/
    └── UAS_1.ipynb                 # Notebook utama


### B. Langkah Menjalankan di Google Colab

1. Buka notebook `UAS_1.ipynb` di Google Colab (upload atau lewat Google Drive).  
2. Mount Google Drive di awal notebook:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

3. Pastikan file train_transaction.csv dan test_transaction.csv berada di folder:

text
/content/drive/MyDrive/ML/

4. Jalankan cell dari atas sampai bawah (Run all). Notebook akan secara otomatis:

- Meload data + optimisasi memori.
- Melakukan preprocessing & feature engineering.
- Melakukan split train–validation.
- Menangani imbalance dengan SMOTE + class weight.
- Melakukan training DNN dengan focal loss.
- Menampilkan evaluasi & visualisasi (loss, AUC, precision, recall, ROC, confusion matrix).
- Melakukan prediksi pada test set & membuat submission.csv.
- Menyimpan model dan artefak ke /content/drive/MyDrive/ML/models/.
- Setelah semua selesai, file submission.csv dapat diunduh dari Google Drive.

5. Kesimpulan Singkat
Telah dibangun pipeline end-to-end untuk fraud detection yang mencakup memory optimization, preprocessing,
feature engineering, handling imbalance, training DNN dengan focal loss, evaluasi metrik utama, dan pembuatan file submission.

Model mencapai ROC-AUC ≈ 0.76 dengan recall fraud ≈ 43%, yang menunjukkan model cukup sensitif terhadap transaksi penipuan
meskipun precision masih rendah (trade-off yang umum pada sistem pendeteksi fraud). Seluruh proses dirancang agar stabil di Google Colab gratis (RAM 12GB),
dengan semua artefak (model, scaler, threshold, fitur, dan submission) tersimpan rapi di Google Drive sehingga mudah dievaluasi maupun digunakan kembali.

6. Identitas
Nama : Eggy Alfan Ananta

Kelas : TK4602

NIM : 1103223194
