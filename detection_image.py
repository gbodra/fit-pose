import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

filename = 'basketball.png'

# Run inference on 'bus.jpg'
results = model(f'imgs/{filename}')  # results list

annotated_img = results[0].plot()

while True:
    cv2.imshow("YOLOv8 Inference", annotated_img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save the annotated image
cv2.imwrite(f'imgs/basketball_annotated.png', annotated_img)