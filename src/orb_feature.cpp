#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;

struct sort_pred {
    bool operator()(const std::pair<float,int> &left, const std::pair<float,int> &right) {
        return left.first > right.first;
    }
};

vector<cv::KeyPoint> ssc(vector<cv::KeyPoint> keyPoints, int numRetPoints,float tolerance, int cols, int rows){
    // several temp expression variables to simplify solution equation
    int exp1 = rows + cols + 2*numRetPoints;
    long long exp2 = ((long long) 4*cols + (long long)4*numRetPoints + (long long)4*rows*numRetPoints + (long long)rows*rows + (long long) cols*cols - (long long)2*rows*cols + (long long)4*rows*cols*numRetPoints);
    double exp3 = sqrt(exp2);
    double exp4 = numRetPoints - 1;

    double sol1 = -round((exp1+exp3)/exp4); // first solution
    double sol2 = -round((exp1-exp3)/exp4); // second solution

    int high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    int low = floor(sqrt((double)keyPoints.size()/numRetPoints));

    int width;
    int prevWidth = -1;

    vector<int> ResultVec;
    bool complete = false;
    unsigned int K = numRetPoints; unsigned int Kmin = round(K-(K*tolerance)); unsigned int Kmax = round(K+(K*tolerance));

    vector<int> result; result.reserve(keyPoints.size());
    while(!complete){
        width = low+(high-low)/2;
        if (width == prevWidth || low>high) { //needed to reassure the same radius is not repeated again
            ResultVec = result; //return the keypoints from the previous iteration
            break;
        }
        result.clear();
        double c = width/2; //initializing Grid
        int numCellCols = floor(cols/c);
        int numCellRows = floor(rows/c);
        vector<vector<bool> > coveredVec(numCellRows+1,vector<bool>(numCellCols+1,false));

        for (unsigned int i=0;i<keyPoints.size();++i){
            int row = floor(keyPoints[i].pt.y/c); //get position of the cell current point is located at
            int col = floor(keyPoints[i].pt.x/c);
            if (coveredVec[row][col]==false){ // if the cell is not covered
                result.push_back(i);
                int rowMin = ((row-floor(width/c))>=0)? (row-floor(width/c)) : 0; //get range which current radius is covering
                int rowMax = ((row+floor(width/c))<=numCellRows)? (row+floor(width/c)) : numCellRows;
                int colMin = ((col-floor(width/c))>=0)? (col-floor(width/c)) : 0;
                int colMax = ((col+floor(width/c))<=numCellCols)? (col+floor(width/c)) : numCellCols;
                for (int rowToCov=rowMin; rowToCov<=rowMax; ++rowToCov){
                    for (int colToCov=colMin ; colToCov<=colMax; ++colToCov){
                        if (!coveredVec[rowToCov][colToCov]) coveredVec[rowToCov][colToCov] = true; //cover cells within the square bounding box with width w
                    }
                }
            }
        }

        if (result.size()>=Kmin && result.size()<=Kmax){ //solution found
            ResultVec = result;
            complete = true;
        }
        else if (result.size()<Kmin) high = width-1; //update binary search range
        else low = width+1;
        prevWidth = width;
    }
    // retrieve final keypoints
    vector<cv::KeyPoint> kp;
    for (unsigned int i = 0; i<ResultVec.size(); i++) kp.push_back(keyPoints[ResultVec[i]]);

    return kp;
}


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

void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                const int numToKeep )
{
        if( keypoints.size() < numToKeep ) { return; }

        //
        // Sort by response
        //
        std::sort( keypoints.begin(), keypoints.end(),
                        [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
                        {
                        return lhs.response > rhs.response;
                        } );

        std::vector<cv::KeyPoint> anmsPts;

        std::vector<double> radii;
        radii.resize( keypoints.size() );
        std::vector<double> radiiSorted;
        radiiSorted.resize( keypoints.size() );

        const float robustCoeff = 1.11; // see paper

        for( int i = 0; i < keypoints.size(); ++i )
        {
                const float response = keypoints[i].response * robustCoeff;
                double radius = std::numeric_limits<double>::max();
                for( int j = 0; j < i && keypoints[j].response > response; ++j )
                {
                        radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
                }
                radii[i]       = radius;
                radiiSorted[i] = radius;
        }

        std::sort( radiiSorted.begin(), radiiSorted.end(),
                        [&]( const double& lhs, const double& rhs )
                        {
                        return lhs > rhs;
                        } );

        const double decisionRadius = radiiSorted[numToKeep];
        for( int i = 0; i < radii.size(); ++i )
        {
                if( radii[i] >= decisionRadius )
                {
                        anmsPts.push_back( keypoints[i] );
                }
        }

        anmsPts.swap( keypoints );
}


int main(){

        // Import image; Instantiate keypoints and descriptor
        //cv::Mat img = cv::imread("/home/aditya/Desktop/Team_Humanoid/SLAM/KITTI/data/0000000000.png");
        cv::Mat img = cv::imread("/home/aditya/Desktop/Team_Humanoid/SLAM/2011_09_26_drive_0002_extract/2011_09_26/2011_09_26_drive_0002_extract/image_00/data/0000000000.png");
        std::vector<cv::KeyPoint> keypoints_anms;
        std::vector<cv::KeyPoint> keypoints_orb;
        std::vector<cv::KeyPoint> keypoints_ssc;
        std::vector<cv::KeyPoint> keypoints_anms2;
        
        cv::Mat descriptor;
        cv::Mat descriptor2;
        // Apply ORB 
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create(15000);
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
        //keypoints_ssc = ssc(keyPointsSorted, 1000, 0.1,img.cols, img.rows);
        adaptiveNonMaximalSuppresion(keypoints_anms2, 1000);

        // Draw Keypoints
        cv::Mat anms_keypoints;
        cv::drawKeypoints(img, keypoints_anms, anms_keypoints);
        
        cv::Mat orb_keypoints;
        cv::drawKeypoints(img, keypoints_orb, orb_keypoints);

        //cv::Mat ssc_keypoints;
        //cv::drawKeypoints(img, keypoints_ssc, ssc_keypoints);

        cv::Mat anms2_keypoints;
        cv::drawKeypoints(img, keypoints_anms2, anms2_keypoints);
        

        // Display resutls
        cv::imshow("ORB Keypoints before ANMS", orb_keypoints);
        cv::waitKey();
        
        //Write results
        cv::imwrite("../results/orbs.png", orb_keypoints);
        cv::imwrite("../results/anms.png", anms_keypoints);
        //cv::imwrite("../results/ssc.png", ssc_keypoints);
        cv::imwrite("../results/anms2.png",anms2_keypoints);

        return 0;

}
