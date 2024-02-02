# fun getAngle(firstPoint: PoseLandmark, midPoint: PoseLandmark, lastPoint: PoseLandmark): Double {
#         var result = Math.toDegrees(atan2(lastPoint.getPosition().y - midPoint.getPosition().y,
#                 lastPoint.getPosition().x - midPoint.getPosition().x)
#                 - atan2(firstPoint.getPosition().y - midPoint.getPosition().y,
#                 firstPoint.getPosition().x - midPoint.getPosition().x))
#         result = Math.abs(result) // Angle should never be negative
#         if (result > 180) {
#             result = 360.0 - result // Always get the acute representation of the angle
#         }
#         return result
#     }
import math


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
