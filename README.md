# automi_SLAM
SLAM module for automi

Running:
To work with it, type 'make' in build directory and then run the executable

Description:
So far, ORB has been implemented on a picture. 5000 keypoints are detected which are sorted bv strength and keypoints within a tolerance distance are eliminated. Finally, 1000 keypoints are of interest and used. This additional processing yields a more uniform distrubution of keypoints. 

What to do next?
1. Do the same for a video
