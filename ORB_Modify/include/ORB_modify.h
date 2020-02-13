#include <iostream>
#include <string>

#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

class ORB_modify
{
public:
    ORB_modify(const string &strSettingPath);

    void ORB_feature(cv::Mat &im);
    void UndistortKeyPoints();

public:
    //ORB
    ORBextractor* mpORBextractor;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    
    cv::Mat mImGray;

    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    std::vector<cv::KeyPoint> mvKeys, mvKeysUn;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;
};