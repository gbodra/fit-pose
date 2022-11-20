import cv2
import calculate_angles
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
incorrect_angle_counter = 0

outdir, inputflnm = './', 'vids/PXL_20221119_130822683.mp4'
# outdir, inputflnm = './', 'vids/PXL_20221119_132147650_2.mp4'

cap = cv2.VideoCapture(inputflnm)

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_annotated_w_angle.mp4'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 40, (frame_width, frame_height))

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.flip(image, 0)

    if not ret:
        break
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # last_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
    angle = calculate_angles.get_angle(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE])

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    position = (50, frame_height - 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    if angle <= 100.0:
        # vermelho
        color = (0, 0, 255)
        incorrect_angle_counter = incorrect_angle_counter + 1
    else:
        # azul
        color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    img_text = 'Angulo Movimento: ' + str(angle) + ' | Movimentos Incorretos: ' + str(incorrect_angle_counter)
    image = cv2.putText(image, img_text, position, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    out.write(image)

pose.close()
cap.release()
out.release()
