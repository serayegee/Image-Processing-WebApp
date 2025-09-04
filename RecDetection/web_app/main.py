import cv2
import processor  # bizim yazdığımız dosya

image = cv2.imread("test.jpg")

binary = processor.binary_threshold(image)
cleaned = processor.clean_noise(binary)
contours = processor.find_contours(cleaned)
annotated = processor.draw_bounding_boxes(image, contours)

cv2.imshow("Sonuç", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
