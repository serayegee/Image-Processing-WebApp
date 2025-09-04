import cv2
import csv
import os

image_path = "results/cropped/frame_00088/otsu_cleaned/crop_1.jpg"
csv_path = "object_areas1.csv"
min_area = 50

def display_image_with_areas(img_path, csv_path, min_area=50):
    if not os.path.exists(img_path):
        print("Görsel bulunamadı.")
        return
     
    # Görseli oku
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Konturları bul
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dosya adı ve crop numarasını al
    base_name = os.path.splitext(os.path.basename(img_path))[0]  # crop_1
    parent_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # frame_00087
    unique_name = f"{parent_name}_{base_name}"  # frame_00087_crop_1

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Alan: {int(area)}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # CSV’ye benzersiz isim + alan kaydet
                writer.writerow([unique_name, int(area)])

    cv2.imshow("Alanlar", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


display_image_with_areas(image_path, csv_path, min_area)
