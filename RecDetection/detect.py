import cv2
from ultralytics import YOLO

# Eğittiğin modeli yükleme
"""model = YOLO("runs/train/metal_shapes/weights/best.pt")"""
model = YOLO("runs/detect/metal_shapes2/weights/best.pt")

# Kamera açma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yapmma
    results = model(frame, conf=0.5)  # conf = güven eşiği

    # Çizimleri frame üzerine uygulama
    annotated_frame = results[0].plot()

    # Rectangle sayısı
    num_rectangles = len(results[0].boxes)
    cv2.putText(annotated_frame, f"Rectangles: {num_rectangles}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0), 2)

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()
