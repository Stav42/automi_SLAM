import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

class Extractor:
    
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.last = None

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def denormalize(self, x,y, shape):
        x_denorm = int(round(x+shape[0]//2))
        y_denorm = int(round(y+shape[1]//2))
        return x_denorm, y_denorm

    def extract_grid(self, img):
        kp, des = self.orb.detectAndCompute(img,None)
        return kp
       
    def extract_spread(self, img):
        
        #detecting
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        
        #extracting
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        #matching
        ret = []
        matches = None
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
                    
            for m, n in matches:
                if m.distance<0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
        
        ret = np.array(ret)
    
        model = None
        #Normalize to centre
        #ret[:, :, 0] -= img.shape[0]//2
        #ret[:, :, 1] -= img.shape[1]//2
    
        #filter
        if len(ret)>0:
            #Normalize to centre
            ret[:, :, 0] -= img.shape[0]//2
            ret[:, :, 1] -= img.shape[1]//2
            
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                    FundamentalMatrixTransform,
                    min_samples = 8,
                    residual_threshold=1,
                    max_trials=50)
            
            ret = ret[inliers]
        
        #Printing the Fundamental Matrix Obtained
        #Single Value Decomposition of the FM
            s,v,d = np.linalg.svd(model.params)
            print(v)
        

        self.last = {'kps': kps, 'des': des}
        return ret
    """
    def extract_anms(self, img):
        kp = self.orb.detect(img)
        kp = sorted(kp, key=lambda x: x.response, reverse=True)
         
        selected_keypoints = ssc(
        kp, 750, 0.1, img.shape[1], img.shape[0]
         )
 
        return selected_keypoints
    """
