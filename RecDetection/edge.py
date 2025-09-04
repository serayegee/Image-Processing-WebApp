import cv2
import os
import pandas as pd
import numpy as np

# Referans frame ölçüleri
ref_w_px, ref_h_px = 390, 537 #frame_00000 için
real_w_cm, real_h_cm = 50,70
pixels_per_cm_x = ref_w_px / real_w_cm
pixels_per_cm_y = ref_h_px / real_h_cm

# Kök klasörler
cropped_root = "results/cropped"
annotated_root = "results/otsu_annotated"
os.makedirs(annotated_root, exist_ok=True)

all_data = []

# Tüm frame klasörlerini tara
for frame_folder in os.listdir(cropped_root):
    frame_path = os.path.join(cropped_root, frame_folder)
    if not os.path.isdir(frame_path):
        continue

    # Sadece otsu_cleaned klasörünü işle
    process_folder = "otsu_cleaned"
    process_path = os.path.join(frame_path, process_folder)
    if not os.path.isdir(process_path):
        continue

    # Annotated klasör yolu
    annotated_process_path = os.path.join(annotated_root, frame_folder, process_folder)
    os.makedirs(annotated_process_path, exist_ok=True)

    # Her görseli işle
    for file in os.listdir(process_path):
        if not file.endswith(".jpg"):
            continue

        img_path = os.path.join(process_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            #Dikdörtgeni bulma
            rect = cv2.minAreaRect(cnt)
            (cx_f, cy_f), (w_f,h_f), angle = rect

            # Merkez noktası
            cx, cy = int(cx_f), int(cy_f)

            # Köşe noktaları
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Görsel üstüne dikdörtgeni çiz
            cv2.drawContours(img, [box], 0, (0,255,0),2)

            # Merkeze daire
            cv2.circle(img, (cx,cy), 5, (0,0,255), -1)

            # Dikdörtgenin kenarları
            v_hor = (box[1]-box[0])
            v_ver = (box[3]-box[0])

            # Çizgiler
            cv2.line(img, (cx - v_hor[0]//2, cy - v_hor[1]//2), (cx + v_hor[0]//2, cy + v_hor[1]//2), (0, 0, 255), 1)
            cv2.line(img, (cx - v_ver[0] // 2, cy - v_ver[1] // 2), (cx + v_ver[0] // 2, cy + v_ver[1] // 2), (0, 0, 255), 1)

            # Pixel-cm
            w_cm = w_f / pixels_per_cm_x
            h_cm = h_f / pixels_per_cm_y
            area_cm2 = w_cm * h_f
            
            # Görsel üzerine metinleri çiz
            # Metin konumunu döndürülmüş dikdörtgenin köşesine göre ayarla
            cv2.putText(img, f"{w_cm:.1f}x{h_cm:.1f} cm", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(img, f"Area:{area_cm2:.1f} cm2", (box[0][0], box[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # CSV verisi
            all_data.append({
                "frame": frame_folder,
                "process": process_folder,
                "img_file": file,
                "object_id": i+1,
                "cx": cx,
                "cy": cy,
                "width_px": w_f,
                "height_px": h_f,
                "angle": angle,
                "area_px": w_f * h_f,
                "width_cm": w_cm,
                "height_cm": h_cm,
                "area_cm2": area_cm2,
                "corners": box.tolist() # NumPy dizisini listeye dönüştür
            })


        # Annotated görseli kaydet
        out_path = os.path.join(annotated_process_path, file.replace(".jpg","_annotated.jpg"))
        cv2.imwrite(out_path, img)
        print("Saved annotated image:", out_path)

# CSV’yi detection klasörüne kaydet
csv_out_path = os.path.join(os.getcwd(), "objects_info.csv")
df = pd.DataFrame(all_data)
df.to_csv(csv_out_path, index=False)
print("CSV saved:", csv_out_path)

