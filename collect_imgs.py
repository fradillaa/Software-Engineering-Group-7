import cv2
import os

# Konfigurasi
DATA_DIR = './data'

number_of_classes = 24
dataset_size = 50  # Gambar yang diambil per sesi/orang

# Membuat folder dataset jika belum ada
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Akses kamera
cap = cv2.VideoCapture(0)

# Label huruf SIBI Alphabet tanpa 'J' dan 'Z'
label_chars = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# number_of_classes = len(label_chars)
for j, char_label in enumerate(label_chars):
    # Buat folder untuk kelas jika belum ada
    class_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    

    # Tampilkan nama huruf, bukan hanya index
    print(f"Collecting data for class '{char_label}' (index {j})")
    
    # Tunggu user tekan 'q' dulu
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Failed to access Webcam")
            break
        
        cv2.putText(frame, f"Pose: {char_label}", (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) == ord('q'):
            break

    # Ambil dan simpan gambar
    # Hitung berapa file yang sudah ada, lanjut dari sana
    existing_files = os.listdir(class_path)
    existing_count = len([f for f in existing_files if f.endswith('.jpg')])

    # Skip kalau sudah penuh (misal 200 gambar)
    if dataset_size >= 200:
        print(f"Class {j} already has 200 images, skipping...")
        continue
    
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Simpan dengan index yang tidak menimpa file lama
        save_index = existing_count + counter
        filename = os.path.join(class_path, f"{save_index}.jpg")
        cv2.imwrite(filename, frame)

        counter += 1

    print(f"Collected {counter} new images for class '{char_label}'\n")

cap.release()
cv2.destroyAllWindows()