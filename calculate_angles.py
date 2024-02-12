import math
import mediapipe as mp


def get_angle(first_point, mid_point, last_point):
    result = math.degrees(math.atan2(last_point.y - mid_point.y, last_point.x - mid_point.x)
                          - math.atan2(first_point.y - mid_point.y, first_point.x - mid_point.x))
    result = abs(result)
    if result > 180:
        result = 360.0 - result

    return round(result, 2)

def shoulder_bending(first_point, last_point):
    if first_point.x < last_point.x:
        return True

    return False

def vertical_landmark(pose_landmarks, frame_width, frame_height):
    mp_pose = mp.solutions.pose
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    if left_hip:
        x = int(left_hip.x * frame_width)
        y = int(left_hip.y * frame_height) + 50
        return x, y

    x = int(right_hip.x * frame_width)
    y = int(right_hip.y * frame_height) + 50
        
    return x, y
