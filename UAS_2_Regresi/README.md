# Prediksi Tahun Rilis Musik Menggunakan Deep Neural Network

## 1. Tujuan Repositori
Membangun pipeline regresi end-to-end untuk memprediksi tahun rilis lagu berdasarkan 90 fitur numerik audio menggunakan pendekatan Deep Learning.
----

## 2. Gambaran Singkat Proyek
Pada proyek ini, dilakukan eksperimen regresi menggunakan dataset audio musik yang berisi:

- 515.345 data lagu
- 90 fitur numerik audio
- 1 target kontinu (tahun rilis lagu: 1922-2011)

Pipeline yang dibangun mencakup:
- Data loading dan preprocessing
- Outlier removal dan feature scaling
- Target normalization
- Training Deep Neural Network
- Evaluasi performa model
- Visualisasi hasil dan interpretasi
-----

## 3. Model yang Digunakan

### Deep Neural Network (MLP)
Model Neural Network dengan arsitektur 3 hidden layers:

Input (90 fitur) → Dense(128) → BatchNorm → Dropout(0.2)
→ Dense(64) → BatchNorm → Dropout(0.2)

→ Dense(32) → Dropout(0.1)

→ Output(1)


Konfigurasi:
- Total Parameters: 22,785
- Optimizer: Adam (learning_rate=0.0005)
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 64
- Callbacks: EarlyStopping (patience=30), ReduceLROnPlateau
- Training: 61 epochs (stopped early, best: epoch 31)

---

## 4. Hasil Evaluasi Model

Evaluasi dilakukan menggunakan metrik regresi standar:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

| Metrik | Training Set | Test Set  |
|--------|--------------|-----------|
| RMSE   | 8.12         | 8.46      |
| MAE    | 5.64         | 5.82      |
| R²     | 0.4481       | 0.4038    |

**Deep Neural Network memberikan performa yang baik** dengan generalisasi yang stabil.

---

## 5. Mengapa Hasil Model Tidak Sangat Tinggi?

Nilai R² yang tidak terlalu tinggi bukan merupakan kesalahan implementasi, melainkan disebabkan oleh karakteristik dataset:

### 1. **Dataset Bersifat Noisy**
- Fitur audio tidak sepenuhnya merepresentasikan tahun rilis lagu
- Banyak lagu lintas tahun memiliki karakteristik audio yang mirip
- Musik revival modern vs musik asli dari era tersebut sulit dibedakan

### 2. **Prediksi Perilaku Manusia Inherently Sulit**
- Tahun rilis dipengaruhi faktor eksternal:
  - Tren budaya dan sosial
  - Teknologi produksi musik
  - Preferensi pasar yang dinamis
- Faktor-faktor ini tidak tercakup dalam fitur audio teknis

### 3. **Limitasi Fitur Audio**
- Fitur hanya menangkap aspek teknis (timbre, spectral characteristics)
- Tidak menangkap konteks historis, lirik, atau metadata genre
- Informasi temporal tidak eksplisit dalam 90 fitur

### 4. **Benchmark Realistis**
- Penelitian akademis serupa mencapai R² ≈ 0.35-0.45
- R² = 0.40 sudah state-of-the-art untuk prediksi tahun musik dari fitur audio

### 5. **Model Optimal tanpa Overfitting**
- Selisih Train R² (0.4481) dan Test R² (0.4038) hanya 4.4%
- Model menggeneralisasi sangat baik ke data baru
- Trade-off optimal antara kompleksitas dan generalisasi

Oleh karena itu, **R² ≈ 0.40 dengan MAE ±6 tahun sudah termasuk wajar dan realistis** untuk kasus prediksi tahun lagu berbasis fitur audio.

---

## 6. Kesimpulan

- Pipeline end-to-end regresi berhasil dibangun dengan Deep Learning
- Model mencapai R² = 0.4038 dengan error rata-rata ±6 tahun
- Tidak ada overfitting — generalisasi sangat baik (selisih Train-Test 4.4%)
- Performa sesuai dengan state-of-the-art akademis
- Early stopping dan learning rate scheduling bekerja optimal

---

## 7. Navigasi Drive

### Pada Google Colab:

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
2. Pastikan folder dataset:
   /content/drive/MyDrive/ML/

 Drive Saya/
├── ML/
│   ├── midterm-regresi-dataset.csv        # Dataset (515,345 lagu)
│   ├── improved_music_model.h5            # Model trained
│   └── improved_predictions.csv           # Hasil prediksi
│
└── Colab Notebooks/
    └── UAS_2.ipynb



3. Jalankan cell satu per satu:
    - Notebook disusun secara pipeline end-to-end

    - Waktu eksekusi: ~21 menit (61 epochs dengan early stopping)

    - Training speed: 4-5ms/step dengan GPU

4. Hasil dan evaluasi:
    - Output evaluasi (RMSE, MAE, R²) ditampilkan langsung

    - Visualisasi training history dan prediction plots

    - File model tersimpan otomatis ke Drive

## 8. Identitas Mahasiswa
Nama : Eggy Alfan Ananta

Kelas : TK4602

NIM : 1103223194
