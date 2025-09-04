import cv2
import os
import random

# Çıktı klasörleri
train_folder = "datasets/metals/images/train"
val_folder = "datasets/metals/images/val"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Video dosyaları
video_files = [
    "videos/v1.mp4",
    "videos/v2.mp4",
    "videos/v3.mp4",
    "videos/v4.mp4"
]

frame_count = 0
frame_skip = 5  # Her 5. frame alınacak

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            # %80 train, %20 val
            if random.random() < 0.8:
                folder = train_folder
            else:
                folder = val_folder

            frame_filename = os.path.join(folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        frame_index += 1

    cap.release()
    print(f"{video_path} işlendi, toplam frame sayısı: {frame_count}")

print(f"Tüm videolar işlendi, toplam {frame_count} frame kaydedildi.")
