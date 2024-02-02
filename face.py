import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


model_path = 'models/face_landmarker.task'

video_path = "vids/face_detection.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('vids/face_output.mp4', fourcc, fps, (frame_width, frame_height))

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ndarray.tobytes(frame_rgb))

        # Calculate the timestamp for the current frame
        frame_timestamp_ms = frame_count * (1000 / fps)

        # Process the image with your landmarker or other processing steps here
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw the face landmarks on the image
        if face_landmarker_result:
            for face_landmarks in face_landmarker_result.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)

        # Write the frame into the file 'output.mp4'
        out.write(frame_rgb)
        
        # Display the resulting frame
        cv2.imshow('Frame', frame_rgb)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment frame count
        frame_count += 1

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
