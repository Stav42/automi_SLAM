import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

H = 480
W = 640


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis =1)

class Extractor:
    
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.last = None

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K  = K
        self.Kinv = np.linalg.inv(self.K)


    def normalize(self, x):
        return np.dot(self.Kinv, add_ones(x).T).T[:,0:2] 

    def denormalize(self, x,y, shape):
        #get this cleared up a bit
        ret = np.dot(self.K, np.array([x,y,1.0]))
     #   ret /= ret[2]

        return int(round(ret[0])), int(round(ret[1]))
      
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
   
        #filter
        if len(ret)>0:

            #Normalize to centre
            ret[:, 0, :] = self.normalize(ret[:,0,:])
            ret[:, 1, :] = self.normalize(ret[:,1,:])

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
