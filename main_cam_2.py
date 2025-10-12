#!source venv/bin/activate
import cv2
from tracker2 import*
import time
from picamera2 import Picamera2

### Parameters
movie = "IMG_5285_short.mov"
run_movie = False
frame_width = 640
frame_height = 480
movie_scale_percent = 50
history_frames = 100
var_threshold = 60
contour_area_thrs = 400 #increase if tiny objects get detected. Decrease if bicyclist are not detected.

obj=cv2.createBackgroundSubtractorMOG2(history=history_frames,varThreshold=var_threshold)
tracker=Tracker2()
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
        t = time.time()
        roi=frame[1:300,1:220]
        mask=obj.apply(roi)
        _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
        cnt,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        points=[]
        for c in cnt:
            area=cv2.contourArea(c)
            if area > contour_area_thrs:
            #            cv2.drawContours(roi,[c],-1,(0,255,0),2)
                x,y,w,h=cv2.boundingRect(c)
                points.append([t,area,x,y,w,h])
        point=tracker.update(points)
        for i in point:
            t, area, x,y,w,h,id = i
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(roi,str(id),(x,y -1),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            print(i) # Diese i enthalten die Daten über Zeit, Grösse und Ort.

        cv2.imshow("FRAME",frame)
        if cv2.waitKey(32)&0xFF==27: # waiting for esc key
            break
except Exception as e:
    print("An error occurred: ", e)
finally:
    cap.stop()
    cv2.destroyAllWindows()
