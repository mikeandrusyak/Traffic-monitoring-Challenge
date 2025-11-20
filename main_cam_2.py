#!source venv/bin/activate
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from tracker2 import *


### PARAMETERS
history_frames = 3000
var_threshold = 16

# Area thresholds (you can fine-tune)
AREA_MIN = 800
AREA_MAX = 40000

# ROI (modify if needed)
ROI_TOP = 170
ROI_BOTTOM = 460
ROI_LEFT = 315
ROI_RIGHT = 530


### BACKGROUND SUBTRACTORS (day/night)
mog_day = cv2.createBackgroundSubtractorMOG2(
    history=400, varThreshold=32, detectShadows=False
)

mog_night = cv2.createBackgroundSubtractorMOG2(
    history=900, varThreshold=18, detectShadows=False
)

kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))


### CAMERA CONFIG
cap = Picamera2()

video_config = cap.create_video_configuration(
    main={"size": (1280,720), "format": "RGB888"},
    lores={"size": (640,480), "format": "YUV420"}   # HIGHER RES
)

cap.configure(video_config)
cap.start()
time.sleep(1)
print("Starting traffic monitoring...")


### TRACKER
tracker = Tracker2()


while True:

    # 1. Capture YUV420 lores frame
    yuv = cap.capture_array("lores")

    # 2. Convert to grayscale
    gray_full = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)

    # 3. Apply ROI
    gray = gray_full[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

    # 4. Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 5. Select day/night model
    brightness = np.mean(gray)
    if brightness > 80:
        mask = mog_day.apply(gray)
    else:
        mask = mog_night.apply(gray)

    # 6. Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)


    # 7. --- MERGE OBJECTS (fixes BUS problem!) ---
    merged_mask = np.zeros_like(mask)

    # Draw all initial bounding boxes into merged_mask
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnt:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(merged_mask, (x,y), (x+w,y+h), 255, -1)

    # Merge nearby objects (bus becomes ONE object)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel_merge)

    # Final contours AFTER merging
    cnt_final, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 8. Collect points for tracker
    points = []
    t = time.time()

    for c in cnt_final:
        area = cv2.contourArea(c)
        if AREA_MIN < area < AREA_MAX:
            x,y,w,h = cv2.boundingRect(c)
            points.append([t, area, x, y, w, h])


    # 9. Track objects
    objects = tracker.update(points)

    # 10. Draw tracked objects
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for (t, area, x, y, w, h, obj_id) in objects:
        cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(display, str(obj_id), (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    # 11. Display windows
    cv2.imshow("MASK", mask)
    cv2.imshow("MERGED", merged_mask)
    cv2.imshow("TRACKER", display)


    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.stop()
cv2.destroyAllWindows()