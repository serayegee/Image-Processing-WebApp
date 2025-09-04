import cv2
import os
import csv

cropped_dir = "results/cropped"
csv_file = "object_areas1.csv"

# CSV başlığı oluştur
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["folder", "process", "img_file", "object_id", "area", "x", "y", "width", "height"])

    # Her kırpılmış fotoğraf klasörünü dolaşma
    for folder_name in os.listdir(cropped_dir):
        folder_path = os.path.join(cropped_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Cleaned klasörlerini bul
        for process_folder in os.listdir(folder_path):
            if not process_folder.endswith("_cleaned"):
                continue

            process_path = os.path.join(folder_path, process_folder)

            for img_file in os.listdir(process_path):
                if not img_file.lower().endswith((".jpg", ".png")):
                    continue

                img_path = os.path.join(process_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Nesneleri bulma
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                print(f"{folder_name}/{process_folder}/{img_file} içindeki cisimler:")
                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area < 50:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)  # kutu bilgisi
                    print(f" Cisim {i+1}: Alan = {area} piksel")
                    writer.writerow([folder_name, process_folder, img_file, i+1, int(area), x, y, w, h])
