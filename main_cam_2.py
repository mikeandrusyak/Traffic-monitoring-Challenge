#!source venv/bin/activate
import cv2
from tracker2 import*
import time
from datetime import datetime
from picamera2 import Picamera2
import os
from dotenv import load_dotenv
import numpy as np
import save_to_database

#Google Cloud SQL
load_dotenv()
engine = save_to_database.connect_with_connector()
session = save_to_database.create_engine_session(engine)

### Parameters
movie = "IMG_5285_short.mov"
run_movie = False
frame_width = 640
frame_height = 480
movie_scale_percent = 50
history_frames = 200
var_threshold = 9
іcontour_area_thrs = 200  # Minimum threshold
max_contour_area = 8000  # Add upper limit
truck_threshold = 3000    # Threshold for trucks



obj=cv2.createBackgroundSubtractorMOG2(history=history_frames,varThreshold=var_threshold)
tracker=Tracker2()
frame_counter = 0 
try:
    cap = Picamera2()
    lsize = (320, 240)
    video_config = cap.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"},
                                                 lores={"size": lsize, "format": "YUV420"})
    cap.configure(video_config)
    #encoder = H264Encoder(1000000)

    cap.start()
    print("starting monitoring...")
    time.sleep(1)

    while True:
        #frame=cap.capture_array()
        frame=cap.capture_array("lores")
        frame_counter += 1 
        t = time.time()
        roi=frame[1:220, 1:320] 
        mask=obj.apply(roi)
        _,mask=cv2.threshold(mask,230,255,cv2.THRESH_BINARY)  # Slightly lower the threshold
        
        # Find all initial contours
        cnt_initial,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new mask for the final result
        final_mask = np.zeros_like(mask)
        
        # Process each contour individually
        for c in cnt_initial:
            area = cv2.contourArea(c)
            
            if area < contour_area_thrs:
                continue  # Ignore too small
                
            # Create a temporary mask for this contour
            temp_mask = np.zeros_like(mask)
            cv2.drawContours(temp_mask, [c], -1, 255, -1)
            
            if area > truck_threshold:
                # For large objects - aggressive morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
            else:
                # For small objects - gentle morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
            
            # Add the processed contour to the final mask
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        # Find final contours from the processed mask
        cnt,_=cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        points=[]
        for c in cnt:
            area=cv2.contourArea(c)
            if contour_area_thrs < area < max_contour_area:
                x,y,w,h=cv2.boundingRect(c)
                # Additional aspect ratio check
                aspect_ratio = w/h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Reasonable ratio
                    points.append([t,area,x,y,w,h])
        point=tracker.update(points)
        for i in point:
            t, area, x,y,w,h,id = i
            
            # Determine vehicle type and color
            if area > truck_threshold:
                vehicle_type = "TRUCK"
                color = (0,255,255)  # Yellow
            else:
                vehicle_type = "CAR"
                color = (0,0,255)    # Red
                
            cv2.rectangle(roi,(x,y),(x+w,y+h),color,2)
            cv2.putText(roi,f"{vehicle_type}:{id}",(x,y-1),cv2.FONT_HERSHEY_COMPLEX,0.6,color,1)
            readable_time = datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Frame:{frame_counter}, {vehicle_type} ID:{id}, Area:{area}, Pos:({x},{y}), Size:({w},{h}), Time:{readable_time}")
            # Insert data into Supabase
            # try:
                 # save_to_database.saveData(engine, frame_counter, id, area, x, y, w, h, t)
            # except Exception as e:
            #     print(f"Database error: {e}")

        # Display all windows for debugging
        cv2.imshow("MASK", final_mask)    # Shows the processed binary mask
        cv2.imshow("ROI", roi)            # Shows the region of interest with tracking boxes
        cv2.imshow("FRAME", frame)        # Shows the full camera frame
        
        if cv2.waitKey(32)&0xFF==27: # waiting for esc key
            break
except Exception as e:
    print("An error occurred: ", e)
finally:
    cap.stop()
    cv2.destroyAllWindows()
