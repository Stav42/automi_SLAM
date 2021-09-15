#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;

struct sort_pred {
    bool operator()(const std::pair<float,int> &left, const std::pair<float,int> &right) {
        return left.first > right.first;
    }
};

std::vector<cv::KeyPoint> brownANMS(std::vector<cv::KeyPoint> keyPoints, int numRetPoints) {
        std::vector<std::pair<float,int> > results;
        results.push_back(std::make_pair(FLT_MAX,0));
        for (unsigned int i = 1;i<keyPoints.size();++i){ //for every keyPoint we get the min distance to the previously visited keyPoints
                float minDist = FLT_MAX;
                for (unsigned int j=0;j<i;++j){
                        float exp1 = (keyPoints[j].pt.x-keyPoints[i].pt.x);
                        float exp2 = (keyPoints[j].pt.y-keyPoints[i].pt.y);
                        float curDist = std::sqrt(exp1*exp1+exp2*exp2);
                        minDist = std::min(curDist,minDist);
                }
                results.push_back(std::make_pair(minDist,i));
        }
        std::sort(results.begin(),results.end(),sort_pred()); //sorting by radius
        std::vector<cv::KeyPoint> kp;
        for (int i=0;i<numRetPoints;++i) kp.push_back(keyPoints[results[i].second]); //extracting numRetPoints keyPoints

        return kp;
}


int main(){

        //Load Video
        cv::VideoCapture cap("/home/aditya/Desktop/Team_Humanoid/SLAM/automi_SLAM/data/2.mp4");

        if(!cap.isOpened()){
            cout<<"Error in opening";
            return -1;
        }

        while(1){
        
                cv::Mat img1;
                cap.read(img1);
                //cv::cvtColor(img1, img1, CV_BGR2GRAY); 
                if(img1.empty())break;
                //cout<<"Channels of video frame is: "<<img1.channels()<<endl;
                // Import image; Instantiate keypoints and descriptor
                //cv::Mat img = cv::imread("/home/aditya/Desktop/Team_Humanoid/SLAM/KITTI/data/0000000000.png");
            
                cv::Mat img = cv::imread("/home/aditya/Desktop/Team_Humanoid/SLAM/2011_09_26_drive_0002_extract/2011_09_26/2011_09_26_drive_0002_extract/image_00/data/0000000000.png");
                //std::vector<cv::KeyPoint> keypoints_anms;
                cout<<"Channels of working image is: "<<img.channels();
                std::vector<cv::KeyPoint> keypoints_orb;
                std::vector<cv::KeyPoint> keypoints_anms2;
                std::vector<cv::KeyPoint> keypoints_anms;

                cv::Mat descriptor;
                cv::Mat descriptor2;
                // Apply ORB 
                cv::Ptr<cv::Feature2D> orb = cv::ORB::create(10000);
                orb->detectAndCompute(img, cv::Mat(), keypoints_orb, descriptor);
                orb->detectAndCompute(img, cv::Mat(), keypoints_anms2, descriptor2);


                // Sorting keypoints by decreasing order of strength
                vector<float> responseVector;
                for (unsigned int i =0 ; i<keypoints_orb.size(); i++) responseVector.push_back(keypoints_orb[i].response);
                vector<int> Indx(responseVector.size()); std::iota (std::begin(Indx), std::end(Indx), 0);
                cv::sortIdx(responseVector, Indx, CV_SORT_DESCENDING);
                vector<cv::KeyPoint> keyPointsSorted;
                for (unsigned int i = 0; i < keypoints_orb.size(); i++) keyPointsSorted.push_back(keypoints_orb[Indx[i]]);


                // Apply ANMS algorithm for more uniform results
                //adaptiveNonMaximalSuppresion(keypoints,10);
                keypoints_anms = brownANMS(keyPointsSorted, 1000);

                // Draw Keypoints
                cv::Mat anms_keypoints;
                cv::drawKeypoints(img, keypoints_anms, anms_keypoints);
                
                // Display the result
                //cv::imshow("Frame", anms_keypoints);
                cv::imshow("Frame", img1);
                char c=(char)cv::waitKey(25);
                cout<<img1;
	            if(c==27)
                break;
               // cv::imwrite("../new_anms.png", anms_keypoints);
        }

        cap.release();
        cv::destroyAllWindows();
        return 0;

}
