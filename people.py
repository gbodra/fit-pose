import cv2
import time
from ultralytics import YOLO
from collections import Counter

# Record the start time
start_time = time.time()

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video file
video_path = "vids/StartSe_Test_1.mp4"
cap = cv2.VideoCapture(video_path)

names = model.names

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
out = cv2.VideoWriter('vids/StartSe_Test_1_out.mp4', fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    # Initialize the Counter object inside the loop
    object_counts = Counter()

    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # results = model(frame)
        results = model.predict(frame, classes=[0], conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        for c in results[0].boxes.cls:
            object_name = names[int(c)]
            object_counts[object_name] += 1

        # Display the count of 'person' detections as an annotation on the video
        if 'person' in object_counts:
            cv2.putText(annotated_frame, f"Count Person: {object_counts['person']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("People Counter", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

# Record the end time
end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time

print(f"The script took {runtime} seconds to run.")