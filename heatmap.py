from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO("yolov8n.pt")
video = 'Castello_Branco_2'
# cap = cv2.VideoCapture(f"vids/{video}.mp4")
cap = cv2.VideoCapture("http://200.246.220.24:8081/webview/wms?source=7&command=start_video&format=JPEG_COLOR&quality=100&ax_disabled&grapherr&start_video_on_start_up")
# http://200.246.220.24:8081/webview/wms?source=17&command=start_video&format=JPEG_COLOR&quality=100&ax_disabled&grapherr&start_video_on_start_up

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter(f"vids/{video}_heatmap.mp4", cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, (w, h))

# Init heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=w,
                     imh=h,
                     view_img=True,
                     shape="circle")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    video_writer.write(im0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()