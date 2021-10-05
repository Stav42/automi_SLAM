import cv2
import time
import numpy as np
from ssc import ssc
from extractor import Extractor
    
A = Extractor()

def process_frames(frame):

    kps, pts, matches = A.extract_spread(frame)
    print(matches)
    img2 = cv2.drawKeypoints(frame, kps, outImage = None, color=(0,255,0), flags=0)
    cv2.imshow('Frame', img2)



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
