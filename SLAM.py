import cv2
import time
import numpy as np

class Feature_Extractor:
    
    def __init__(self):
        self.orb = cv2.ORB_create()

    def extract_grid(self, img):
        N = 20
        kp, des = self.orb.detectAndCompute(img,None)
        img2 = cv2.drawKeypoints(img,kp,outImage = None, color=(0,255,0), flags=0)
        cv2.imshow('Frame', img2)
        #for i in range(0,N-1):
        #    frame = img[24*i:24*(i+1),32*i:32*(i+1)]
        #    kp, des = orb.detectAndCompute(frame,None)
        #    print("I'm working")

    def extract_spread(self, img):
        pass

A = Feature_Extractor()

def process_frames(frame):
    A.extract_grid(frame)
    #cv2.imshow('Frame', frame)

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
