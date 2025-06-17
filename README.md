# 📌 Nama Proyek

Mimico: Bridging Communication Gaps with Intelligent Sign Language Translation

## 📁 Struktur Folder

| Folder/File             | Fungsi |
|-------------------------|--------|
| `app/`                  | Interface web |
| `collect_imgs.py`       | Script untuk mengumpulkan gesture tangan dari webcam |
| `create_dataset.py`     | Mengolah data mentah menjadi dataset siap pakai |
| `train_classifier.py`   | Melatih model klasifikasi berdasarkan dataset |
| `inference_classifier.py` | Prediksi huruf dari gambar input menggunakan model |
| `model.p`               | File model hasil pelatihan (format Pickle) |
| `data.pickle`           | Data encode hasil pelabelan dataset |
| `requirements.txt`      | Daftar library Python yang digunakan |
| `images/`               | Berisi gambar untuk tampilan backgroung website dan kamus isyarat |
| `data/`                 | Dataset lokal (tidak disertakan di repo karena besar) |
| `test/`                 | Folder pengujian sistem |
| `myvenv/`               | Virtual environment lokal (tidak diunggah) |

## 📊 Dataset

Dataset ini dibuat sendiri untuk keperluan proyek akhir.

- Jumlah data: 300 gambar isyarat tangan
- Format: `.jpg` dalam folder sesuai label huruf A–Z (kecuali J dan Z)
- Ukuran total: ±1.3GB
- Format struktur folder: `dataset/A/img001.jpg`, dst.

Karena ukuran terlalu besar untuk GitHub, dataset bisa diunduh di:

🔗 [Download via One Drive](https://binusianorg-my.sharepoint.com/personal/faradilla_chandra_binus_ac_id/_layouts/15/guestaccess.aspx?share=Etw5WZhnwkFCmsDRU5IaNVUBsPt20fydzsPE2nDYm2kvmg&e=kewAXT)

