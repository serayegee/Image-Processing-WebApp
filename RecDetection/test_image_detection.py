import cv2
from ultralytics import YOLO
import os

# Eğittiğin modeli yükle
model = YOLO("runs/detect/metal_shapes2/weights/best.pt")

# Test edilecek görüntüyü yükle
image_path = "datasets/metals/images/train/frame_00000.jpg"
image = cv2.imread(image_path)

# Model ile tahmin yap
results = model(image, conf=0.5)

# Tespitleri çiz
annotated_image = results[0].plot()

# Tespit edilen rectangle sayısını ekrana yaz
num_rectangles = len(results[0].boxes)
cv2.putText(annotated_image, f"Rectangles: {num_rectangles}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Görüntüyü göster
cv2.imshow("YOLOv8 Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sonucu kaydet
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path=os.path.join(output_dir, "frame_00000_annotated.jpg")
cv2.imwrite(output_path, annotated_image)
print(f"Annotated image saved: {output_path}")

# Her görüntü için ayrı bir alt klasör
image_name = os.path.splitext(os.path.basename(image_path))[0]  # frame_00087

# Her dikdörtgeni kırpılmış şekilde kaydetme
cropped_dir = os.path.join(output_dir, "cropped", image_name)
os.makedirs(cropped_dir, exist_ok=True)

for i, box in enumerate(results[0].boxes.xyxy):
    x1,y1,x2,y2 = map(int,box)
    cropped = image[y1:y2, x1:x2]
    crop_path = os.path.join(cropped_dir, f"crop_{i+1}.jpg")
    cv2.imwrite(crop_path, cropped)
    print(f"Cropped image saved: {crop_path}")
