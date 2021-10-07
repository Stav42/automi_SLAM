import cv2
import time
import numpy as np
from ssc import ssc
from extractor import Extractor


F = 400
H = 480
W = 640

K = np.array(([F, 0, W//2],[0, F, H//2],[0, 0, 1]))

A = Extractor(K)


def process_frames(frame):

    matches = A.extract_spread(frame)
    if matches is None:
        return
    
    for m in matches:
        u1, v1 = m[0]
        u2, v2 = m[1]
        
        u1, v1 = A.denormalize(u1,v1, frame.shape)
        u2, v2 = A.denormalize(u2, v2, frame.shape)

        cv2.circle(frame, (int(u1), int(v1)),  3, (0, 255, 0), -1)
        cv2.circle(frame, (int(u2), int(v2)),  3, (0, 255, 0), -1)
        cv2.line(frame, (int(u1), int(v1)), (int(u2), int(v2)), (0, 255, 0), thickness=3, lineType=8)

    cv2.imshow('Frame', frame)



if __name__ == "__main__":

    cap = cv2.VideoCapture('/home/aditya/Desktop/Team_Humanoid/SLAM/automi_SLAM_v2/automi_SLAM/data/video/gazebo.mp4')
    flag = 1
    height = 0
    width =0
    col = 0

    while(cap.isOpened()):
        
        ret, frame = cap.read()

        if(flag):
            height, width, col = frame.shape
            flag = 0
        
        process_frames(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
       
    cap.release()
    cv2.destroyAllWindows()
