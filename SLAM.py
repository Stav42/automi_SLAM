import cv2
import time
import numpy as np
from ssc import ssc

class Feature_Extractor:
    
    def __init__(self):
        self.orb = cv2.ORB_create()

    def extract_grid(self, img):
        N = 20
        kp, des = self.orb.detectAndCompute(img,None)
        return kp
        #img2 = cv2.drawKeypoints(img,kp,outImage = None, color=(0,255,0), flags=0)
        #cv2.imshow('Frame', img2)
        #kp = []
        #des = []
        #for i in range(0,N-1):
        #    for j in range(0, N-1):
        #        frame = img[24*i:24*(i+1),32*j:32*(j+1)]
        #        kp_new, des_new = self.orb.detectAndCompute(frame,None)
        #        kp.append(kp_new)
        #        des.append(des_new)
        
        #img2 = cv2.drawKeypoints(img,kp,outImage = None, color=(0,255,0), flags=0)
        #cv2.imshow('Frame', img2)

    def extract_spread(self, img):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 7000, qualityLevel=0.01, minDistance=3)
        return feats

    def extract_anms(self, img):
        kp = self.orb.detect(img)
        kp = sorted(kp, key=lambda x: x.response, reverse=True)
         
        selected_keypoints = ssc(
        kp, 750, 0.1, img.shape[1], img.shape[0]
         )

        return selected_keypoints
     
A = Feature_Extractor()

def process_frames(frame):

    pts = A.extract_spread(frame)
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = A.orb.compute(frame, kps)
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
