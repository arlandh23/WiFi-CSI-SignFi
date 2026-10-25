# WiFi CSI Gesture Recognition (Track A) — SignFi

Pipeline: CSI kompleks (Intel 5300, 3x30, 200 paket/gestur) -> **Extraction** (amp & fase) -> **Filtering** (DWT + Butterworth LPF)
-> **Phase sanitization** (unwrap + transformasi linier) -> **Feature structuring** (tensor 2x90x200 & spektrogram PCA-STFT) -> Deep Learning.

## Struktur
```
src/preprocessing.py   # tahap 1-4 preprocessing CSI
src/models.py          # Baseline 1D-CNN, CNN-BiLSTM-Attention, ResNet-2D (STFT)
src/trainer.py         # training loop, evaluasi, latency
notebook.ipynb         # pipeline lengkap: data -> training -> evaluasi -> ablation -> demo
outputs/               # figur, tabel hasil, checkpoint, config
```

## Cara menjalankan
1. `pip install -r requirements.txt`
2. Unduh `dataset_lab_276_dl.mat` dari https://wm1693.box.com/s/z9vsrn3998n4xyzkpqtj89yclk28eatp (SignFi, Ma et al. 2018)
3. Buka `notebook.ipynb` di Google Colab (GPU) -> Run all.

## Hasil (test set, 276 kelas)
| Model                    |   Params |   Test Acc (top-1) |   Test Top-5 |   F1 (macro) |
|:-------------------------|---------:|-------------------:|-------------:|-------------:|
| Baseline 1D-CNN          |   610580 |             0.9903 |       1      |       0.9898 |
| Proposed CNN-BiLSTM-Attn |   961557 |             0.9891 |       1      |       0.9878 |
| ResNet-2D (STFT)         |   344404 |             0.2935 |       0.5483 |       0.2839 |

Model terbaik: **Baseline 1D-CNN**.
