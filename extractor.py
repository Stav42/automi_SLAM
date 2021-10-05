import cv2
import numpy as np


class Extractor:
    
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.last = None


        index_params = dict( algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

        self.flann = cv2.FlannBasedMatcher(index_params, dict(checks=100))


    def extract_grid(self, img):
        N = 20
        kp, des = self.orb.detectAndCompute(img,None)
        return kp
       
    def extract_spread(self, img):
        
        #detecting
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        
        #extracting
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        #matching
        matches = None
        if self.last is not None:
            matches = self.flann.knnMatch(des, self.last['des'], k=2)

        self.last = {'kps ': kps, 'des': des}
        return kps, des, matches


    def extract_anms(self, img):
        kp = self.orb.detect(img)
        kp = sorted(kp, key=lambda x: x.response, reverse=True)
         
        selected_keypoints = ssc(
        kp, 750, 0.1, img.shape[1], img.shape[0]
         )
 
        return selected_keypoints
 
