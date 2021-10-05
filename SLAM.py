import cv2
import time
import numpy as np
from ssc import ssc
from extractor import Extractor
    
A = Extractor()

def process_frames(frame):

    #kps, pts, matches = A.extract_spread(frame)
    matches = A.extract_spread(frame)
    if matches is None:
        return
    
    print(len(matches))
    #u1, v1 = [kps1.pt[0] for kps1, kps2 in matches],[kps1.pt[1] for kps1, kps2 in matches]
    #u2, v2 = [kps2.pt[0] for kps1, kps2 in matches],[kps2.pt[1] for kps1, kps2 in matches]
    for i in range(len(matches[0])):
        #print(u1, v1)
        u1, v1 = matches[0][i].pt[0], matches[0][i].pt[1]
        u2, v2 = matches[1][i].pt[0], matches[1][i].pt[1]

        print(u1,v1)
        cv2.circle(frame, (int(u1), int(v1)),  3, (0, 255, 0), -1)
        cv2.circle(frame, (int(u2), int(v2)),  3, (0, 255, 0), -1)
        cv2.line(frame, (int(u1), int(v1)), (int(u2), int(v2)), (0, 255, 0), thickness=3, lineType=8)

    #img2 = cv2.drawKeypoints(frame, kps, outImage = None, color=(0,255,0), flags=0)
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
