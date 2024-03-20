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
    hip_left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    ear_right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    ear_left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

    if hip_left:
        point_left = (int(hip_left.x * frame_width), int(hip_left.y * frame_height))
    
    if hip_right:
        point_right = (int(hip_right.x * frame_width), int(hip_right.y * frame_height))

    if ear_left:
        point_left_top = (int(ear_left.x * frame_width), int(ear_left.y * frame_height))

    if ear_right:
        point_right_top = (int(ear_right.x * frame_width), int(ear_right.y * frame_height))

    x_median = (point_left[0] + point_right[0]) / 2
    # point_1 = (int(x_median), point_left[1] - 100)
    point_1 = (int(x_median), point_left_top[1])
    point_2 = (int(x_median), point_left[1] + 100)
    
    return point_1, point_2
