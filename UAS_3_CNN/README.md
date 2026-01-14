# Klasifikasi Gambar Ikan Menggunakan CNN dan Transfer Learning

## 1. Tujuan Repositori
Membangun pipeline klasifikasi gambar end-to-end untuk mengidentifikasi 31 spesies ikan menggunakan Custom CNN dan Transfer Learning (MobileNetV2).

---

## 2. Gambaran Singkat Proyek
Dataset yang digunakan:
- **Total gambar:** 13,312 (Training: 8,801 | Validation: 2,751 | Test: 1,760)
- **Jumlah kelas:** 31 spesies ikan
- **Image size:** 224×224 pixels (RGB)

Pipeline mencakup:
- Data preprocessing dengan augmentasi intensif
- Handling class imbalance menggunakan class weights (ratio 11.11x)
- Training Custom CNN baseline (4 conv blocks)
- Transfer Learning MobileNetV2 (Phase 1: Feature Extraction + Phase 2: Fine-Tuning)
- Auto-checkpoint system untuk resume training
- Evaluasi dan visualisasi hasil

---

## 3. Model yang Digunakan

### Custom CNN (Baseline)
Arsitektur: 4 convolutional blocks dengan BatchNorm, Dropout, dan L2 regularization

Input (224×224×3) → Conv(32) → Conv(64) → Conv(128) → Conv(256)

→ Dense(512) → Dense(31, softmax)

 - Epochs: 20
 - Optimizer: Adam (lr=0.001)
 - Training time: 0.11 min (dengan checkpoint)

### Transfer Learning - MobileNetV2

**Phase 1 (25 epochs):** Feature extraction dengan base model frozen  
**Phase 2 (15 epochs):** Fine-tuning top 30 layers (lr=0.0001)

MobileNetV2 (pre-trained) → GlobalAvgPool → Dense(512) → Dense(256) → Dense(31)

---

## 4. Hasil Evaluasi Model

| Model              | Test Accuracy | Precision | Recall | F1-Score |
|--------------------|---------------|-----------|--------|----------|
| Custom CNN         | 43.07%        | 0.4688    | 0.4307 | 0.4128   |
| Transfer Learning  | **93.75%**    | 0.9443    | 0.9375 | 0.9385   |

**Improvement:** Transfer Learning unggul **+50.68%** dibanding baseline.

**Top performing classes (Transfer Learning):**
 - Gold Fish: 98.80% F1
 - Gourami: 98.39% F1
 - Long-Snouted Pipefish: 98.08% F1
 - Green Spotted Puffer, Scat Fish, Silver Carp: 100% F1

---

## 5. Fitur Utama Pipeline

### Auto-Checkpoint System
- Smart checkpoint setiap 5 epoch dengan metadata
- Auto-resume training dari epoch terakhir
- Menyimpan history untuk visualisasi lengkap

### Class Imbalance Handling
- Dataset imbalance ratio: 11.11x
- Class weights: min 0.23 (kelas besar) hingga max 2.58 (kelas kecil)

### Data Augmentation
Training augmentation: rotation (±40°), shift (±30%), zoom (±30%), flip, brightness adjustment

---

## 6. Mengapa Transfer Learning Unggul?

- **Pre-trained features:** MobileNetV2 sudah belajar dari 1.4M gambar ImageNet
- **Two-phase strategy:** Feature extraction + fine-tuning mencegah overfitting
- **Efficient architecture:** Cocok untuk Google Colab free GPU
- **Strong generalization:** Train-test gap sangat kecil

---

## 7. Navigasi Drive
```text
Struktur folder:
Drive Saya/ML/
├── checkpoints/ # Folder checkpoint models
│ ├── best_custom_cnn.h5
│ ├── best_transfer_final.h5
│ ├── best_transfer_phase1.h5
│ ├── custom_cnn_epoch_010.h5
│ ├── custom_cnn_epoch_015.h5
│ ├── custom_cnn_epoch_020.h5
│ ├── custom_cnn_history.json
│ ├── custom_cnn_metadata.json
│ ├── transfer_finetune_epoch_010.h5
│ ├── transfer_finetune_epoch_015.h5
│ ├── transfer_finetune_history.json
│ ├── transfer_finetune_metadata.json
│ ├── transfer_phase1_epoch_010.h5
│ ├── transfer_phase1_epoch_015.h5
│ ├── transfer_phase1_epoch_020.h5
│ ├── transfer_phase1_epoch_025.h5
│ ├── transfer_phase1_history.json
│ └── transfer_phase1_metadata.json
│
├── train/ # 8,801 images (31 classes)
│ ├── Bangus/
│ ├── Big Head Carp/
│ ├── Black Spotted Barb/
│ ├── ...
│ └── Tilapia/
│
├── val/ # 2,751 images (31 classes)
│ └── (same 31 class folders)
│
├── test/ # 1,760 images (31 classes)
│ └── (same 31 class folders)
│
├── class_indices.json # Class name mapping
├── evaluation_results.json # Evaluation metrics
├── fish_classification_final.h5 # Final model (HDF5 format)
└── fish_classification_final.keras # Final model (Keras format)

```
Path di Colab:
```python
base_path = '/content/drive/MyDrive/ML'
checkpoint_dir = '/content/drive/MyDrive/ML/checkpoints'
train_dir = '/content/drive/MyDrive/ML/train'
val_dir = '/content/drive/MyDrive/ML/val'
test_dir = '/content/drive/MyDrive/ML/test'
```
## 8. Cara Menjalankan
 - Mount Google Drive di Colab
 - Pastikan GPU aktif (Runtime > T4 GPU)
 - Jalankan notebook cell by cell
---
## 9. Kesimpulan
 - Pipeline klasifikasi 31 spesies ikan berhasil dibangun
 - Transfer Learning mencapai 93.75% accuracy (excellent untuk 31 kelas)
 - Auto-checkpoint system optimal untuk resume training
 - Class weights mengatasi imbalanced dataset
 - Model siap untuk deployment
---
## 10. Identitas Mahasiswa

Nama: Eggy Alfan Ananta

Kelas: TK4602

NIM: 1103223194
