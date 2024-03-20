import time
import cv2
import calculate_angles
import mediapipe as mp
from datetime import timedelta

start_time = time.time()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
incorrect_angle_counter = 0

outdir, inputflnm = './', 'vids/Braco_1.mp4'

cap = cv2.VideoCapture(inputflnm)

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_angle.mp4'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('H', '2', '6', '4'), 20, (frame_width, frame_height))

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        break
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks is not None:
        angle = calculate_angles.get_angle(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST])

        is_shoulder_bending = calculate_angles.shoulder_bending(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])

        # Plot vertical point
        point_1, point_2 = calculate_angles.vertical_landmark(results.pose_landmarks, frame_width, frame_height)
        # Draw a line between two points
        image = cv2.line(image, point_1, point_2, (0, 255, 255), 10)

        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, frame_height - 50)
        fontScale = 1
        color = (255, 0, 0)

        if angle <= (180*0.8):
            color = (0, 0, 255)
            incorrect_angle_counter = incorrect_angle_counter + 1
        else:
            color = (255, 0, 0)

        thickness = 2
        # img_text = f"Angulo Dir: {angle_r:.2f} | Angulo Esq: {angle_l:.2f}"
        img_text = 'Angulo Movimento: ' + str(angle) + ' | Movimentos Incorretos: ' + str(incorrect_angle_counter)
        
        # Calculate the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(img_text, font, fontScale, thickness)[0]
        box_coords = ((position[0] - 5, position[1] + 5), (position[0] + text_width + 5, position[1] - text_height - 5))
        cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

        image = cv2.putText(image, img_text, position, font, fontScale, color, thickness, cv2.LINE_AA)

    out.write(image)

pose.close()
cap.release()
out.release()

end_time = time.time()
execution_time = end_time - start_time
formatted_time = str(timedelta(seconds=int(execution_time)))
print(f"Execution time: {formatted_time}")
