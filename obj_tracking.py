from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np


#Initializing the YOLO model
model = YOLO('yolov8l.pt')

cap = cv2.VideoCapture('vid5.mp4')

#Initializing a variable for collecting the points of each ID
track_history = defaultdict(lambda: [])

frame_count = 1
while cap.isOpened():

    ret, frame = cap.read()


    if ret:
        results = model.track(frame, conf=0.50, persist=True, tracker="bytetrack.yaml")
        if results[0].boxes.id == None: # This check  is for the frames in which nothing is detected. We ignore that frame and continue to the next frame
            continue
        boxes = results[0].boxes.xywh
        track_ids = results[0].boxes.id.int().tolist()

        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            
            x,y,w,h = box
            track = track_history[track_id] #Here we're collecting all the center points of each ID
            track.append((float(x), float(y)))
            if len(track) > 30: # removing the point at index 0 once we reach 30 points
                track.pop(0)
            
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) # np.array(track, np.int32) will also do the job. These are the points of a particular ID for last 30 frames which we'll be drawing the frames.
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        cv2.imwrite(f'frames_for _making_video/frame{frame_count}.jpg', annotated_frame)
        frame_count += 1
        cv2.imshow('Window', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()